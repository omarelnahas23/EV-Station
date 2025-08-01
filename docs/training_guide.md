# Training Guide

This comprehensive guide walks you through the entire process of fine-tuning Llama 3 7B on EV charging domain data using LoRA/QLoRA techniques.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Environment Setup](#environment-setup)
- [Data Collection](#data-collection)
- [Data Processing](#data-processing)
- [QA Dataset Generation](#qa-dataset-generation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Inference and Deployment](#inference-and-deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Prerequisites

### Knowledge Requirements

- **Python Programming**: Intermediate level familiarity with Python
- **Machine Learning**: Basic understanding of neural networks and language models
- **Command Line**: Comfortable with terminal/command prompt operations
- **Git**: Basic version control operations

### Software Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Git**: For cloning the repository
- **Conda/Pip**: For package management

## Hardware Requirements

### Minimum Configuration (QLoRA)
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti, RTX 3080)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space (SSD recommended)
- **CPU**: 4+ cores

### Recommended Configuration (LoRA)
- **GPU**: 12-16GB VRAM (RTX 4070, RTX 4080, RTX 3080)
- **RAM**: 32GB system memory
- **Storage**: 100GB free SSD space
- **CPU**: 8+ cores

### Optimal Configuration
- **GPU**: 24GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 64GB system memory
- **Storage**: 200GB NVMe SSD
- **CPU**: 16+ cores

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ev-charging-llm-pipeline.git
cd ev-charging-llm-pipeline
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n ev-charging-llm python=3.10
conda activate ev-charging-llm

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional training dependencies
pip install transformers[torch] peft bitsandbytes datasets accelerate wandb
```

### 4. Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 5. Configure Weights & Biases (Optional)

```bash
# For experiment tracking
wandb login
# Enter your API key from https://wandb.ai/settings
```

## Data Collection

### 1. Configure Data Sources

Edit `config.yaml` to enable/disable data sources:

```yaml
domain: 'electric vehicle charging stations'
use_case: 'QA'
base_model: 'meta-llama/Meta-Llama-3-7B-Instruct'
data_dir: 'data'
enable_web_search: true      # Enable DuckDuckGo/Google search
enable_pdf_download: true    # Enable PDF downloads
enable_api_collection: true  # Enable API data collection
```

### 2. Run Data Collection

```bash
cd data_collection
python collect_data.py
```

**Expected Output:**
```
INFO:__main__:Starting enhanced data collection...
INFO:__main__:=== Collecting from Famous EV Charging Datasets ===
INFO:__main__:Collected 2 ACN-Data sample records
INFO:__main__:Collected 2 DOE EV data records
INFO:__main__:=== Downloading PDF Documents ===
INFO:__main__:Downloaded PDF: IEA Global EV Outlook 2023
INFO:__main__:=== Collecting from Web Sources ===
INFO:__main__:Scraped 537074 characters from Wikipedia
INFO:__main__:Data collection completed! Collected 29 items from 18 sources
```

### 3. Verify Collected Data

```bash
# Check collected data
ls data/
# Should show: raw_data.json, collection_summary.json, pdfs/

# View summary
cat data/collection_summary.json
```

## Data Processing

### 1. Clean and Filter Data

```bash
cd ../data_processing
python process_data.py
```

**Expected Output:**
```
INFO:__main__:Processing 29 raw data items
INFO:__main__:Processed 19 data items
```

### 2. Verify Processed Data

```bash
# Check processed data
ls data/
# Should show: processed_data.json, raw_data.json

# Check quality metrics
python -c "
import json
with open('data/processed_data.json', 'r') as f:
    data = json.load(f)
print(f'Processed items: {len(data)}')
print(f'Average length: {sum(len(item[\"text\"]) for item in data) / len(data):.0f} chars')
"
```

## QA Dataset Generation

### 1. Generate Question-Answer Pairs

```bash
python generate_qa_dataset.py
```

**Expected Output:**
```
INFO:__main__:Loaded 19 processed data items
INFO:__main__:Generating QA pairs from 19 data items
INFO:__main__:Generated 54 training samples and 7 evaluation samples
INFO:__main__:QA Dataset Generation Complete!
INFO:__main__:📊 Summary:
INFO:__main__:  - Total QA pairs: 61
INFO:__main__:  - Training samples: 54
INFO:__main__:  - Evaluation samples: 7
```

### 2. Verify QA Dataset

```bash
# Check generated files
ls data/
# Should show multiple formats: train_lora.json, eval_lora.json, etc.

# Examine QA pairs
python -c "
import json
with open('data/train_lora.json', 'r') as f:
    data = json.load(f)
print(f'Training samples: {len(data)}')
print('\nSample QA pair:')
sample = data[0]['text']
print(sample[:500] + '...')
"
```

## Model Training

### 1. Configure Training Parameters

Edit `lora_config.yaml` for your hardware:

```yaml
# For 8GB GPU (QLoRA)
qlora_config:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"

training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

```yaml
# For 16GB+ GPU (LoRA)
qlora_config:
  load_in_4bit: false

training_args:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

### 2. Start Training

```bash
python train_llama3_lora.py
```

**Expected Output:**
```
INFO:__main__:Starting Llama 3 7B LoRA/QLoRA training for EV Charging QA
INFO:__main__:Loading model and tokenizer...
INFO:__main__:Using 4-bit quantization (QLoRA)
INFO:__main__:Setting up LoRA...
trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.6225
INFO:__main__:Loading and preparing dataset...
INFO:__main__:Training dataset size: 54
INFO:__main__:Evaluation dataset size: 7
INFO:__main__:Starting training...
```

### 3. Monitor Training Progress

#### Using Weights & Biases

If configured, view progress at: https://wandb.ai/your-username/llama3-ev-charging

#### Using Terminal Output

```
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 1.8567, 'learning_rate': 0.00018, 'epoch': 1.0}
{'eval_loss': 1.2345, 'eval_runtime': 12.34, 'epoch': 1.0}
```

#### Using TensorBoard (Alternative)

```bash
# In another terminal
tensorboard --logdir models/llama3-7b-ev-charging-lora/logs
# Open http://localhost:6006
```

### 4. Training Time Estimates

| Configuration | Hardware | Estimated Time |
|---------------|----------|----------------|
| QLoRA | RTX 4060 Ti (8GB) | 3-4 hours |
| QLoRA | RTX 4080 (16GB) | 2-3 hours |
| LoRA | RTX 4090 (24GB) | 1.5-2 hours |
| LoRA | A100 (40GB) | 1-1.5 hours |

## Model Evaluation

### 1. Automated Testing

```bash
python inference_llama3.py --model_path ../models/llama3-7b-ev-charging-lora --test
```

**Expected Output:**
```
🧪 Testing EV Charging Expert Model
==================================================

🔍 Test 1: What are the different types of EV charging connectors?
----------------------------------------
🤖 Response: The main EV charging connector types include Type 1 (SAE J1772) used in North America...

🔍 Test 2: How long does it take to charge an electric vehicle?
----------------------------------------
🤖 Response: EV charging time depends on several factors including battery capacity...
```

### 2. Interactive Testing

```bash
python inference_llama3.py --model_path ../models/llama3-7b-ev-charging-lora --interactive
```

**Example Session:**
```
🚗⚡ EV Charging Expert Chatbot
Ask me anything about electric vehicle charging!
Type 'quit' to exit.

You: What is CHAdeMO?
Assistant: CHAdeMO is a Japanese DC fast charging standard that enables rapid charging of electric vehicles...

You: How does smart charging work?
Assistant: Smart charging optimizes when and how electric vehicles charge to benefit both users and the grid...
```

### 3. Performance Metrics

#### Quantitative Metrics

```python
# Calculate perplexity
from transformers import AutoTokenizer
import torch

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss)

# Usage
perplexity = calculate_perplexity(model, tokenizer, test_text)
print(f"Perplexity: {perplexity:.2f}")
```

#### Qualitative Assessment

Rate responses on:
- **Accuracy**: Technical correctness
- **Completeness**: Comprehensive coverage
- **Clarity**: Easy to understand
- **Relevance**: Appropriate to question

## Inference and Deployment

### 1. Model Loading

```python
from data_processing.inference_llama3 import EVChargingChatbot

# Load the fine-tuned model
chatbot = EVChargingChatbot("models/llama3-7b-ev-charging-lora")

# Generate responses
response = chatbot.generate_response("What are EV charging levels?")
print(response)
```

### 2. Batch Processing

```python
questions = [
    "What are the types of EV connectors?",
    "How does DC fast charging work?",
    "What is Vehicle-to-Grid technology?"
]

for question in questions:
    response = chatbot.generate_response(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

### 3. API Deployment

Create a simple FastAPI server:

```python
# deploy_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from data_processing.inference_llama3 import EVChargingChatbot

app = FastAPI()
chatbot = EVChargingChatbot("models/llama3-7b-ev-charging-lora")

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 512

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    response = chatbot.generate_response(
        request.question, 
        max_new_tokens=request.max_tokens
    )
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

**Error**: `CUDA out of memory`

**Solutions:**
```yaml
# Reduce batch size
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

# Enable gradient checkpointing
hardware:
  use_gradient_checkpointing: true

# Use QLoRA instead of LoRA
qlora_config:
  load_in_4bit: true
```

#### 2. Slow Training

**Symptoms**: Very slow training speed

**Solutions:**
```yaml
# Increase batch size if memory allows
training_args:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

# Use mixed precision
training_args:
  bf16: true

# Reduce sequence length
max_seq_length: 1024
```

#### 3. Poor Model Quality

**Symptoms**: Irrelevant or incorrect responses

**Solutions:**
```yaml
# Increase training epochs
training_args:
  num_train_epochs: 5

# Adjust learning rate
training_args:
  learning_rate: 1e-4

# Add more training data
# Run data collection with more sources
```

#### 4. Model Loading Issues

**Error**: `FileNotFoundError` or model loading failures

**Solutions:**
```bash
# Verify model path
ls models/llama3-7b-ev-charging-lora/

# Check for required files
# Should contain: adapter_config.json, adapter_model.bin, etc.

# Re-run training if files are missing
python train_llama3_lora.py
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in code
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

### GPU Monitoring

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Or use more detailed monitoring
pip install gpustat
watch -n 1 gpustat
```

## Advanced Topics

### 1. Multi-GPU Training

For multiple GPUs, use accelerate:

```bash
# Configure accelerate
accelerate config

# Run training with accelerate
accelerate launch train_llama3_lora.py
```

### 2. Custom Data Sources

Add new data sources by extending the collection pipeline:

```python
def collect_custom_data(config):
    """Collect data from custom sources."""
    data_items = []
    
    # Your custom collection logic here
    custom_data = fetch_from_custom_api()
    
    for item in custom_data:
        data_items.append({
            'text': item['content'],
            'metadata': {
                'source': 'Custom Source',
                'type': 'custom_data',
                'url': item.get('url', '')
            }
        })
    
    return data_items

# Add to main collection function
def collect_data(config):
    # ... existing code ...
    
    # Add custom data collection
    custom_data = collect_custom_data(config)
    data.extend(custom_data)
    
    # ... rest of function ...
```

### 3. Model Merging

Merge LoRA adapters back into the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "models/llama3-7b-ev-charging-lora")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("models/llama3-7b-ev-charging-merged")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
tokenizer.save_pretrained("models/llama3-7b-ev-charging-merged")
```

### 4. Hyperparameter Tuning

Use Weights & Biases sweeps for hyperparameter optimization:

```yaml
# sweep.yaml
program: train_llama3_lora.py
method: bayes
metric:
  goal: minimize
  name: eval_loss
parameters:
  learning_rate:
    values: [1e-4, 2e-4, 5e-4]
  lora_r:
    values: [8, 16, 32]
  lora_alpha:
    values: [16, 32, 64]
```

```bash
wandb sweep sweep.yaml
wandb agent sweep_id
```

### 5. Model Quantization

Post-training quantization for deployment:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "models/llama3-7b-ev-charging-merged",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Best Practices

### 1. Data Quality

- **Validate Sources**: Ensure data sources are authoritative and up-to-date
- **Content Review**: Manually review a sample of generated QA pairs
- **Domain Coverage**: Ensure balanced coverage across EV charging topics
- **Length Distribution**: Monitor question and answer length distributions

### 2. Training Optimization

- **Learning Rate Scheduling**: Use warmup and decay for stable training
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Checkpointing**: Save regular checkpoints during long training runs
- **Evaluation Frequency**: Balance evaluation frequency with training speed

### 3. Resource Management

- **Memory Monitoring**: Keep track of GPU memory usage
- **Storage Planning**: Plan for model checkpoints and dataset storage
- **Backup Strategy**: Regular backups of trained models and data
- **Version Control**: Track configuration changes and model versions

### 4. Quality Assurance

- **Automated Testing**: Regular automated evaluation on benchmark questions
- **Human Evaluation**: Periodic human review of model responses
- **Performance Tracking**: Monitor key metrics over time
- **Error Analysis**: Analyze failure cases to improve the system

## Next Steps

After completing this guide, you might want to:

1. **Experiment with different architectures** (Mistral, CodeLlama)
2. **Add multimodal capabilities** (images, diagrams)
3. **Implement retrieval-augmented generation** (RAG)
4. **Deploy to production** with proper monitoring and scaling


For more information, see:
- [API Reference](api_reference.md)
- [Data Sources Documentation](data_sources.md)


---

*Happy training! 🚗⚡* 