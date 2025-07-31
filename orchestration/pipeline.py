import yaml
import sys
import os
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load pipeline configuration."""
    config_path = Path(__file__).parent / '..' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_data_collection(config):
    """Execute data collection phase."""
    logger.info("Starting data collection phase...")
    try:
        sys.path.append(str(Path(__file__).parent / '..' / 'data_collection'))
        from collect_data import collect_data
        
        data = collect_data(config)
        logger.info(f"Data collection completed. Collected {len(data)} items.")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def run_data_processing(config):
    """Execute data processing phase."""
    logger.info("Starting data processing phase...")
    try:
        sys.path.append(str(Path(__file__).parent / '..' / 'data_processing'))
        from process_data import process_data
        
        processed_data = process_data(config)
        logger.info(f"Data processing completed. Processed {len(processed_data)} items.")
        return True
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return False

def run_dataset_generation(config):
    """Execute dataset generation phase."""
    logger.info("Starting dataset generation phase...")
    try:
        sys.path.append(str(Path(__file__).parent / '..' / 'dataset_generation'))
        from generate_dataset import generate_dataset
        
        train_data, eval_data = generate_dataset(config)
        logger.info(f"Dataset generation completed. Train: {len(train_data)}, Eval: {len(eval_data)} examples.")
        return True
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return False

def run_fine_tuning(config):
    """Execute fine-tuning phase."""
    logger.info("Starting fine-tuning phase...")
    try:
        sys.path.append(str(Path(__file__).parent / '..' / 'fine_tuning'))
        from train_model import train_model
        
        model, tokenizer = train_model(config)
        logger.info("Fine-tuning completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return False

def run_evaluation(config):
    """Execute evaluation phase."""
    logger.info("Starting evaluation phase...")
    try:
        sys.path.append(str(Path(__file__).parent / '..' / 'evaluation'))
        from evaluate_model import evaluate_model
        
        results = evaluate_model(config)
        logger.info("Model evaluation completed successfully.")
        
        # Log key metrics
        ft_coverage = results['fine_tuned']['avg_coverage_score']
        baseline_coverage = results['baseline']['avg_coverage_score']
        improvement = ft_coverage - baseline_coverage
        
        logger.info(f"Coverage improvement: {improvement:+.3f}")
        
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if required directories exist
    required_dirs = ['data_collection', 'data_processing', 'dataset_generation', 
                    'fine_tuning', 'evaluation', 'deployment']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"Required directory not found: {dir_name}")
            return False
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        logger.error("Configuration file not found: config.yaml")
        return False
    
    logger.info("All prerequisites met.")
    return True

def run_full_pipeline(skip_training=False, skip_evaluation=False):
    """Run the complete pipeline end-to-end."""
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("STARTING FULL PIPELINE EXECUTION")
    logger.info("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Exiting.")
        return False
    
    # Load configuration
    config = load_config()
    logger.info(f"Pipeline configuration loaded for domain: {config['domain']}")
    
    # Create necessary directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['fine_tuned_model_dir']), exist_ok=True)
    
    # Phase 1: Data Collection
    if not run_data_collection(config):
        logger.error("Pipeline failed at data collection phase.")
        return False
    
    # Phase 2: Data Processing
    if not run_data_processing(config):
        logger.error("Pipeline failed at data processing phase.")
        return False
    
    # Phase 3: Dataset Generation
    if not run_dataset_generation(config):
        logger.error("Pipeline failed at dataset generation phase.")
        return False
    
    # Phase 4: Fine-tuning (optional skip for testing)
    if not skip_training:
        if not run_fine_tuning(config):
            logger.error("Pipeline failed at fine-tuning phase.")
            return False
    else:
        logger.info("Skipping fine-tuning phase as requested.")
    
    # Phase 5: Evaluation (optional skip)
    if not skip_evaluation and not skip_training:
        if not run_evaluation(config):
            logger.error("Pipeline failed at evaluation phase.")
            return False
    else:
        logger.info("Skipping evaluation phase.")
    
    # Pipeline completion
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("="*60)
    
    return True

def run_data_pipeline_only():
    """Run only the data collection and processing phases."""
    logger.info("Running data pipeline only (collection + processing + dataset generation)...")
    
    config = load_config()
    
    # Create necessary directories
    os.makedirs(config['data_dir'], exist_ok=True)
    
    success = (run_data_collection(config) and 
              run_data_processing(config) and 
              run_dataset_generation(config))
    
    if success:
        logger.info("Data pipeline completed successfully.")
    else:
        logger.error("Data pipeline failed.")
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["full", "data-only", "no-training"], 
                       default="data-only", help="Pipeline execution mode")
    parser.add_argument("--skip-evaluation", action="store_true", 
                       help="Skip evaluation phase")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        success = run_full_pipeline(skip_evaluation=args.skip_evaluation)
    elif args.mode == "no-training":
        success = run_full_pipeline(skip_training=True, skip_evaluation=True)
    else:  # data-only
        success = run_data_pipeline_only()
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed!")
        sys.exit(1) 