import json
import yaml
import os
import logging
from transformers import pipeline, AutoTokenizer
import random
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_text_into_chunks(text: str, tokenizer, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split a long string into chunks of approximately chunk_size tokens."""
    tokens = tokenizer.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

def generate_qa_pairs_from_texts(
    texts: List[str],
    model_name: str,
    max_answer_length: int = 128,
    device: Optional[str] = None,
    target_domain: str = ""
) -> List[Dict[str, str]]:
    """Generate question-answer pairs from text chunks using LLM."""
    logger.info("Initializing QA generation pipeline with model %s", model_name)
    
    try:
        qa_generator = pipeline("text2text-generation", model=model_name, device=device)
    except Exception as e:
        logger.warning("Failed to load %s, falling back to default model: %s", model_name, e)
        qa_generator = pipeline("text2text-generation", model="google/flan-t5-base", device=device)
    
    qa_pairs = []
    
    # Domain-specific question templates for EV charging
    if "electric" in target_domain.lower() or "vehicle" in target_domain.lower():
        question_templates = [
            "What are the different types of electric vehicle charging connectors?",
            "How long does it take to charge an electric vehicle?",
            "What is the difference between AC and DC charging?",
            "What factors affect electric vehicle charging speed?",
            "How does smart charging work?",
            "What is the purpose of a charging management system?",
            "What are the power levels for EV charging?",
            "How does weather affect EV charging?",
            "What is vehicle-to-grid technology?",
            "What are the costs associated with EV charging infrastructure?"
        ]
    else:
        question_templates = [
            "What is the main topic discussed in this text?",
            "What are the key concepts mentioned?",
            "How does this technology work?",
            "What are the benefits mentioned?",
            "What challenges are described?"
        ]
    
    for i, text in enumerate(texts):
        if len(text.strip()) < 100:  # Skip very short texts
            continue
            
        logger.info("Generating QA pairs for text chunk %d/%d", i + 1, len(texts))
        
        # Generate questions based on the text content
        for template in question_templates[:3]:  # Limit to 3 questions per chunk
            try:
                # Create a context-aware prompt
                if "{" in template:
                    question_prompt = template.format(text=text[:500])
                else:
                    question_prompt = f"Based on this text about {target_domain}: {text[:500]}\n\nQuestion: {template}"
                
                # Generate question
                question_response = qa_generator(
                    f"Generate a specific question: {question_prompt}",
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                question = question_response[0]['generated_text'].strip()
                
                # Generate answer based on the text
                answer_prompt = f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"
                answer_response = qa_generator(
                    answer_prompt,
                    max_length=max_answer_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                answer = answer_response[0]['generated_text'].strip()
                
                # Clean up the generated text
                if question.startswith("Question: "):
                    question = question[10:]
                if answer.startswith("Answer: "):
                    answer = answer[8:]
                
                if question and answer and len(question) > 10 and len(answer) > 10:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": text
                    })
                    
            except Exception as e:
                logger.warning("Error generating QA pair for template '%s': %s", template, e)
                continue
    
    logger.info("Generated %d QA pairs from %d text chunks", len(qa_pairs), len(texts))
    return qa_pairs

def build_dataset(
    collected: List[Dict[str, Any]],
    base_model: str,
    chunk_size: int,
    max_answer_length: int,
    target_domain: str,
) -> List[Dict[str, str]]:
    """Build training dataset from collected data."""
    logger.info("Building dataset from %d collected items", len(collected))
    
    # Initialize tokenizer for chunking
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.warning("Failed to load tokenizer for %s: %s", base_model, e)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    all_texts = []
    
    # Extract and chunk texts
    for item in collected:
        content = item["content"]
        if isinstance(content, str):
            # Regular text content
            chunks = split_text_into_chunks(content, tokenizer, chunk_size)
            all_texts.extend(chunks)
        elif isinstance(content, list):
            # Structured data - convert to text
            for record in content:
                if isinstance(record, dict):
                    text_content = " ".join(str(v) for v in record.values() if v)
                    chunks = split_text_into_chunks(text_content, tokenizer, chunk_size)
                    all_texts.extend(chunks)
    
    logger.info("Created %d text chunks for QA generation", len(all_texts))
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs_from_texts(
        texts=all_texts[:50],  # Limit for demonstration
        model_name=base_model,
        max_answer_length=max_answer_length,
        target_domain=target_domain
    )
    
    # Convert to training format
    examples = []
    for qa in qa_pairs:
        example = {
            "question": qa["question"],
            "answer": qa["answer"],
            "context": qa["context"]
        }
        examples.append(example)
    
    logger.info("Built dataset with %d examples", len(examples))
    return examples

def create_training_examples(qa_pairs, domain):
    """Convert QA pairs into training format."""
    examples = []
    
    for qa in qa_pairs:
        # Create instruction-following format
        instruction = f"You are an expert on {domain}. Answer the following question based on the provided context."
        
        input_text = f"Context: {qa['context']}\n\nQuestion: {qa['question']}"
        output_text = qa['answer']
        
        example = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        examples.append(example)
    
    return examples

def generate_dataset(config):
    """Main dataset generation pipeline."""
    # Load processed data
    processed_data_path = os.path.join(config['data_dir'], 'processed_data.json')
    
    if not os.path.exists(processed_data_path):
        logger.warning("Processed data not found. Running data processing first...")
        import sys
        sys.path.append('../data_processing')
        from process_data import process_data
        processed_data = process_data(config)
    else:
        with open(processed_data_path, 'r') as f:
            processed_data = json.load(f)
    
    # Convert processed data to the format expected by build_dataset
    collected_data = []
    for item in processed_data:
        collected_data.append({
            "content": item['text'],
            "type": item['metadata'].get('type', 'text'),
            "source": item['metadata'].get('source', 'unknown')
        })
    
    # Build dataset using the updated function
    examples = build_dataset(
        collected=collected_data,
        base_model=config.get('base_model', 'google/flan-t5-base'),
        chunk_size=config.get('chunk_size', 512),
        max_answer_length=config.get('max_answer_length', 128),
        target_domain=config.get('domain', 'general')
    )
    
    # Convert to training format
    training_examples = create_training_examples(examples, config['domain'])
    
    # Split into train/eval
    random.shuffle(training_examples)
    split_idx = int(0.8 * len(training_examples))
    
    train_data = training_examples[:split_idx]
    eval_data = training_examples[split_idx:]
    
    # Save datasets
    os.makedirs(config['data_dir'], exist_ok=True)
    
    with open(config['training_data_file'], 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(config['eval_data_file'], 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    logger.info(f"Generated {len(train_data)} training examples and {len(eval_data)} evaluation examples")
    return train_data, eval_data

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_dataset(config) 