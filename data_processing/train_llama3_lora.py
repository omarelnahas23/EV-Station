#!/usr/bin/env python3
"""
Training script for Llama 3 7B with LoRA/QLoRA on EV Charging QA dataset
"""

import os
import json
import yaml
import torch
import logging
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="lora_config.yaml"):
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer with quantization."""
    model_name = config['model_name']
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup quantization config for QLoRA
    qlora_config = config.get('qlora_config', {})
    if qlora_config.get('load_in_4bit', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', 'bfloat16'))
        )
        logger.info("Using 4-bit quantization (QLoRA)")
    else:
        quantization_config = None
        logger.info("Using full precision (LoRA)")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
    )
    
    # Prepare model for k-bit training if using quantization
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, config):
    """Setup LoRA configuration."""
    lora_config = config['lora_config']
    
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def load_and_prepare_dataset(config, tokenizer):
    """Load and prepare the QA dataset."""
    train_path = config['train_dataset']
    eval_path = config['eval_dataset']
    
    # Load datasets
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    def tokenize_function(examples):
        """Tokenize the text data."""
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=config.get('max_seq_length', 2048),
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def setup_training_args(config):
    """Setup training arguments."""
    training_config = config['training_args']
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        evaluation_strategy=training_config['evaluation_strategy'],
        eval_steps=training_config['eval_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        dataloader_num_workers=training_config['dataloader_num_workers'],
        remove_unused_columns=training_config['remove_unused_columns'],
        group_by_length=training_config['group_by_length'],
        report_to=training_config.get('report_to', None),
        run_name=training_config.get('run_name', 'llama3-ev-charging'),
        gradient_checkpointing=config.get('hardware', {}).get('use_gradient_checkpointing', True),
        ddp_find_unused_parameters=config.get('hardware', {}).get('ddp_find_unused_parameters', False),
    )
    
    return training_args

def main():
    """Main training function."""
    logger.info("Starting Llama 3 7B LoRA/QLoRA training for EV Charging QA")
    
    # Load configuration
    config = load_config()
    
    # Initialize wandb if configured
    if config['training_args'].get('report_to') == 'wandb':
        try:
            wandb.init(
                project="llama3-ev-charging",
                name=config['training_args']['run_name'],
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    model = setup_lora(model, config)
    
    # Load and prepare dataset
    logger.info("Loading and preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training config
    with open(os.path.join(training_args.output_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main() 