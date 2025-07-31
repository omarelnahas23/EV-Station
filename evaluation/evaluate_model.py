import json
import yaml
import os
import time
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download("punkt", quiet=True)
except Exception:
    pass

def compute_bleu(references: List[str], predictions: List[str]) -> float:
    """Compute average BLEU-4 score across a set of references and predictions."""
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, pred in zip(references, predictions):
        try:
            ref_tokens = nltk.word_tokenize(ref.lower())
            pred_tokens = nltk.word_tokenize(pred.lower())
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Error computing BLEU for reference '{ref[:50]}...': {e}")
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0

def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Compute average ROUGE-1, ROUGE-2 and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    for ref, pred in zip(references, predictions):
        try:
            r = scorer.score(ref, pred)
            for key in scores:
                scores[key] += r[key].fmeasure
        except Exception as e:
            logger.warning(f"Error computing ROUGE for reference '{ref[:50]}...': {e}")
            
    n = len(references)
    if n > 0:
        for key in scores:
            scores[key] /= n
    return scores

def evaluate_model(
    model_path: str,
    eval_examples: List[Dict[str, str]],
    base_model_name: str,
    metrics: List[str],
) -> Dict[str, float]:
    """Evaluate a fine-tuned model against evaluation examples."""
    logger.info("Loading model from %s for evaluation", model_path)
    
    try:
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Create pipeline
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
    except Exception as e:
        logger.warning("Failed to load fine-tuned model, using base model: %s", e)
        generator = pipeline(
            "text2text-generation",
            model=base_model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    # Generate predictions
    logger.info("Generating predictions for %d examples", len(eval_examples))
    predictions = []
    references = []
    
    for example in eval_examples:
        try:
            # Format input
            if "context" in example:
                input_text = f"Context: {example['context']}\n\nQuestion: {example['question']}"
            else:
                input_text = f"Question: {example['question']}"
            
            # Generate prediction
            result = generator(
                input_text,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            prediction = result[0]['generated_text'].strip()
            predictions.append(prediction)
            references.append(example['answer'])
            
        except Exception as e:
            logger.warning("Error generating prediction for question '%s': %s", example['question'][:50], e)
            predictions.append("")
            references.append(example['answer'])
    
    # Compute metrics
    results = {}
    
    if "bleu" in metrics:
        bleu_score = compute_bleu(references, predictions)
        results["bleu"] = bleu_score
        logger.info("BLEU score: %.4f", bleu_score)
    
    if "rouge" in metrics:
        rouge_scores = compute_rouge(references, predictions)
        results.update(rouge_scores)
        logger.info("ROUGE scores: %s", rouge_scores)
    
    # Domain-specific evaluation for EV charging
    domain_score = compute_domain_coverage(references, predictions)
    results["domain_coverage"] = domain_score
    logger.info("Domain coverage score: %.4f", domain_score)
    
    return results

def measure_latency(model_path: str, sample_questions: List[str]) -> Dict[str, float]:
    """Measure inference latency and throughput."""
    logger.info("Measuring latency with %d sample questions", len(sample_questions))
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        logger.warning("Failed to load model for latency measurement: %s", e)
        return {"avg_latency_ms": 0.0, "throughput_qps": 0.0}
    
    latencies = []
    
    for question in sample_questions:
        start_time = time.time()
        try:
            result = generator(
                f"Question: {question}",
                max_length=100,
                num_return_sequences=1
            )
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        except Exception as e:
            logger.warning("Error measuring latency for question '%s': %s", question[:50], e)
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0  # Questions per second
    else:
        avg_latency = 0.0
        throughput = 0.0
    
    return {
        "avg_latency_ms": avg_latency,
        "throughput_qps": throughput
    }

def compute_domain_coverage(references: List[str], predictions: List[str]) -> float:
    """Compute domain-specific coverage score for EV charging domain."""
    ev_keywords = [
        "charging", "charger", "electric", "vehicle", "battery", "connector", "station",
        "power", "voltage", "current", "energy", "kwh", "ac", "dc", "level", "fast",
        "slow", "plug", "socket", "tesla", "chademo", "ccs", "j1772", "grid", "smart"
    ]
    
    total_coverage = 0.0
    
    for ref, pred in zip(references, predictions):
        ref_lower = ref.lower()
        pred_lower = pred.lower()
        
        # Count domain keywords in reference and prediction
        ref_keywords = sum(1 for keyword in ev_keywords if keyword in ref_lower)
        pred_keywords = sum(1 for keyword in ev_keywords if keyword in pred_lower)
        
        # Calculate coverage (intersection over union style)
        if ref_keywords > 0:
            coverage = min(pred_keywords / ref_keywords, 1.0)
        else:
            coverage = 1.0 if pred_keywords == 0 else 0.0
            
        total_coverage += coverage
    
    return total_coverage / len(references) if references else 0.0

def generate_domain_questions():
    """Generate domain-specific evaluation questions for EV charging."""
    return [
        "What are the different types of EV charging connectors?",
        "How long does it take to charge an electric vehicle?",
        "What is the difference between AC and DC charging?",
        "What factors affect electric vehicle charging speed?",
        "How does smart charging work?",
        "What is the purpose of a charging management system?",
        "What are the main power levels for EV charging?",
        "How does weather affect EV charging performance?",
        "What is vehicle-to-grid (V2G) technology?",
        "What are the costs associated with EV charging infrastructure?",
        "How do different connector types affect charging compatibility?",
        "What safety features are built into EV charging stations?",
        "How does load balancing work in charging networks?",
        "What are the benefits of workplace charging?",
        "How do charging sessions get monitored and billed?"
    ]

def evaluate_model_comprehensive(config):
    """Comprehensive model evaluation pipeline."""
    logger.info("Starting comprehensive model evaluation")
    
    # Load evaluation data
    eval_data_path = config.get('eval_data_file', 'data/eval_dataset.json')
    
    if os.path.exists(eval_data_path):
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
    else:
        logger.warning("Evaluation data not found, using domain questions")
        eval_data = []
        domain_questions = generate_domain_questions()
        for question in domain_questions:
            eval_data.append({
                "question": question,
                "answer": "This is a placeholder answer for evaluation.",
                "context": ""
            })
    
    # Convert eval data format if needed
    eval_examples = []
    for item in eval_data:
        if isinstance(item, dict):
            if 'question' in item and 'answer' in item:
                eval_examples.append(item)
            elif 'input' in item and 'output' in item:
                # Convert from instruction format
                question = item['input'].split('Question: ')[-1] if 'Question: ' in item['input'] else item['input']
                context = item['input'].split('Context: ')[-1].split('\n\nQuestion:')[0] if 'Context: ' in item['input'] else ''
                eval_examples.append({
                    "question": question,
                    "answer": item['output'],
                    "context": context
                })
    
    if not eval_examples:
        logger.error("No valid evaluation examples found")
        return {}
    
    # Evaluate the model
    model_path = config.get('fine_tuned_model_dir', 'models/fine_tuned')
    base_model = config.get('base_model', 'google/flan-t5-base')
    metrics = config.get('metrics', ['bleu', 'rouge'])
    
    results = evaluate_model(
        model_path=model_path,
        eval_examples=eval_examples[:20],  # Limit for demo
        base_model_name=base_model,
        metrics=metrics
    )
    
    # Measure latency
    sample_questions = [ex['question'] for ex in eval_examples[:10]]
    latency_results = measure_latency(model_path, sample_questions)
    results.update(latency_results)
    
    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    results_path = 'evaluation_results/evaluation_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed. Results saved to %s", results_path)
    logger.info("Final results: %s", results)
    
    return results

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    evaluate_model_comprehensive(config) 