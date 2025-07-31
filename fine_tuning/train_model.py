import json
import yaml
import os
import logging
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_lora_model(
    base_model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_4bit: bool = False,
):
    """Load a pretrained seq2seq model and apply LoRA adapters."""
    logger.info("Loading base model: %s", base_model_name)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if use_4bit else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v", "k", "o", "wi", "wo"]  # T5-specific modules
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error("Failed to prepare LoRA model: %s", e)
        raise

def preprocess_examples(examples: List[Dict[str, str]], tokenizer, max_length: int = 512):
    """Preprocess training examples for seq2seq format."""
    inputs = []
    targets = []
    
    for example in examples:
        # Format input
        if "context" in example:
            input_text = f"Context: {example['context']}\n\nQuestion: {example['question']}"
        else:
            input_text = f"Question: {example['question']}"
        
        inputs.append(input_text)
        targets.append(example['answer'])
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_lora_model(
    examples: List[Dict[str, str]],
    base_model_name: str,
    output_dir: Path,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    use_4bit: bool = False,
) -> Path:
    """Fine-tune a model using LoRA."""
    logger.info("Starting LoRA fine-tuning with %d examples", len(examples))
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_lora_model(
        base_model_name, lora_r, lora_alpha, lora_dropout, use_4bit
    )
    
    # Preprocess data
    logger.info("Preprocessing training data")
    processed_data = preprocess_examples(examples, tokenizer)
    
    # Create dataset
    train_dataset = Dataset.from_dict(processed_data)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save the model
    logger.info("Saving fine-tuned model to %s", output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def format_instruction(example):
    """Format training examples into instruction format."""
    if example.get('context'):
        return f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
    else:
        return f"Question: {example['question']}\n\nAnswer: {example['answer']}"

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the training examples."""
    texts = [format_instruction(ex) for ex in examples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # For seq2seq, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def setup_lora_model(model, lora_config):
    """Setup LoRA configuration for efficient fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', ["q", "v", "k", "o"])
    )
    
    model = get_peft_model(model, lora_config)
    return model

def train_model(config):
    """Main training function for backward compatibility."""
    logger.info("Starting model training with config")
    
    # Load training data
    with open(config['training_data_file'], 'r') as f:
        train_data = json.load(f)
    
    # Convert to the format expected by train_lora_model
    examples = []
    for item in train_data:
        example = {
            "question": item.get('input', '').replace('Question: ', ''),
            "answer": item.get('output', ''),
            "context": item.get('input', '').split('Context: ')[-1].split('\n\nQuestion:')[0] if 'Context: ' in item.get('input', '') else ''
        }
        examples.append(example)
    
    # Set up output directory
    output_dir = Path(config.get('fine_tuned_model_dir', 'models/fine_tuned'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    model_path = train_lora_model(
        examples=examples,
        base_model_name=config.get('base_model', 'google/flan-t5-base'),
        output_dir=output_dir,
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.05),
        num_train_epochs=config.get('num_train_epochs', 1),
        learning_rate=config.get('learning_rate', 5e-5),
        batch_size=config.get('batch_size', 2),
    )
    
    logger.info(f"Training completed. Model saved to {model_path}")
    return model_path

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_model(config) 