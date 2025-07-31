# EV Charging QA Dataset for Llama 3 7B Fine-tuning

This directory contains the complete pipeline for generating QA datasets from EV charging data and fine-tuning Llama 3 7B with LoRA/QLoRA.

## 📊 Dataset Overview

### Generated Datasets
- **Total QA Pairs**: 61 
- **Training Samples**: 54
- **Evaluation Samples**: 7
- **Topics Covered**: 5 main categories
- **Sources Used**: 9 different data sources

### Topics Distribution
- **User Behavior**: 33 pairs (54%)
- **Infrastructure**: 13 pairs (21%) 
- **Connector Types**: 7 pairs (11%)
- **Charging Levels**: 4 pairs (7%)
- **Smart Charging**: 4 pairs (7%)

### Data Sources
- Research Literature (9 samples)
- Domain Expert Content (12 samples)
- DOE EV Data Collection (6 samples)
- Chinese High-Resolution Dataset (6 samples)
- NREL Alternative Fuel Stations (6 samples)
- Hamburg Public Charging (6 samples)
- ACN-Data Caltech (6 samples)
- Workplace Charging Dataset (6 samples)
- Specialized QA Generation (4 samples)

## 🗃️ File Structure

```
data_processing/
├── data/
│   ├── train_lora.json          # LoRA training data (Llama 3 format)
│   ├── eval_lora.json           # LoRA evaluation data
│   ├── train_dataset.json       # Standard JSON training data
│   ├── eval_dataset.json        # Standard JSON evaluation data
│   ├── train_dataset.jsonl      # JSONL format for streaming
│   ├── eval_dataset.jsonl       # JSONL format for streaming
│   ├── train_dataset_hf/        # Hugging Face Dataset format
│   ├── eval_dataset_hf/         # Hugging Face Dataset format
│   └── qa_dataset_summary.json  # Dataset statistics
├── generate_qa_dataset.py       # QA generation script
├── train_llama3_lora.py         # LoRA/QLoRA training script
├── inference_llama3.py          # Inference and testing script
├── lora_config.yaml             # LoRA/QLoRA configuration
└── process_data.py              # Data preprocessing script
```

## 🚀 Quick Start

### 1. Generate QA Dataset
```bash
cd data_processing
python generate_qa_dataset.py
```

### 2. Train Llama 3 with LoRA
```bash
# Install required packages
pip install transformers peft bitsandbytes datasets accelerate wandb

# Start training
python train_llama3_lora.py
```

### 3. Test the Model
```bash
# Run automated tests
python inference_llama3.py --model_path ../models/llama3-7b-ev-charging-lora --test

# Interactive chat
python inference_llama3.py --model_path ../models/llama3-7b-ev-charging-lora --interactive
```

## 📋 QA Format

### Llama 3 Chat Format
Each QA pair follows the Llama 3 instruction format:

```xml
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert on electric vehicle charging technology and infrastructure. Provide accurate, detailed, and helpful information about EV charging.<|eot_id|><|start_header_id|>user<|end_header_id|>

What are the different types of EV charging connectors?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The main EV charging connector types include: Type 1 (SAE J1772) used in North America and Japan for AC charging up to 7.4kW; Type 2 (IEC 62196) European standard for AC charging up to 22kW; CHAdeMO Japanese DC fast charging standard up to 50kW+; CCS (Combined Charging System) that combines AC and DC charging in one connector; Tesla Supercharger proprietary standard for high-speed DC charging; and GB/T Chinese national standard for both AC and DC charging.<|eot_id|>
```

### Question Categories

#### 1. Connector Types
- Different charging connector standards
- Regional variations and compatibility
- Technical specifications

#### 2. Charging Levels
- Power levels and charging times
- AC vs DC charging
- Infrastructure requirements

#### 3. Infrastructure Planning
- Location considerations
- Technical requirements
- Economic factors

#### 4. Smart Charging
- Grid integration
- Load balancing
- V2G technology

#### 5. User Behavior
- Charging patterns
- Usage statistics
- Adoption factors

## ⚙️ LoRA Configuration

### Model Settings
- **Base Model**: meta-llama/Meta-Llama-3-7B-Instruct
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target Modules**: All attention and MLP layers
- **Dropout**: 0.1

### Training Settings
- **Epochs**: 3
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 2048
- **Precision**: bfloat16

### QLoRA (4-bit) Settings
- **Quantization**: NF4
- **Double Quantization**: Enabled
- **Compute Dtype**: bfloat16

## 💾 Memory Requirements

### LoRA (Full Precision)
- **GPU Memory**: ~14-16GB
- **Recommended**: RTX 4090, A100, H100

### QLoRA (4-bit Quantization)
- **GPU Memory**: ~8-10GB  
- **Recommended**: RTX 3090, RTX 4080, A6000

## 📈 Training Monitoring

### Weights & Biases Integration
Set up W&B tracking:
```bash
wandb login
# Training will automatically log to wandb
```

### Metrics Tracked
- Training/Validation Loss
- Learning Rate Schedule
- GPU Memory Usage
- Training Speed (tokens/sec)

## 🧪 Sample Questions for Testing

```python
test_questions = [
    "What are the different types of EV charging connectors?",
    "How long does it take to charge an electric vehicle?", 
    "What factors should be considered when planning EV charging infrastructure?",
    "What is smart charging and how does it work?",
    "How do EV charging costs compare to gasoline?",
    "What is the difference between AC and DC charging?",
    "How does weather affect EV charging?",
    "What are the benefits of workplace charging?",
    "How does Vehicle-to-Grid (V2G) technology work?",
    "What are the main challenges in EV charging infrastructure deployment?"
]
```

## 🔧 Customization

### Adding New Questions
Modify `question_templates` in `EVChargingQAGenerator` class:

```python
self.question_templates = {
    'new_topic': [
        "New question template 1?",
        "New question template 2?",
    ]
}
```

### Adjusting Answer Length
Modify `extract_relevant_content()` parameters:

```python
def extract_relevant_content(self, text: str, max_length: int = 500):
    # Adjust max_length as needed
```

### Training Configuration
Edit `lora_config.yaml` to modify:
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters
- Hardware settings

## 📊 Quality Metrics

### Answer Quality
- **Average Question Length**: 58.7 characters
- **Average Answer Length**: 341.2 characters
- **Content Relevance**: High (domain-specific sources)
- **Technical Accuracy**: Verified against expert content

### Coverage Analysis
- ✅ Connector standards and compatibility
- ✅ Charging infrastructure planning  
- ✅ User behavior and adoption patterns
- ✅ Smart charging and grid integration
- ✅ Technical specifications and requirements

## 🛠️ Troubleshooting

### Common Issues

#### Out of Memory Error
- Reduce batch size in `lora_config.yaml`
- Enable QLoRA (4-bit quantization)
- Use gradient checkpointing

#### Slow Training
- Increase batch size if memory allows
- Use multiple GPUs with DataParallel
- Enable mixed precision (bf16)

#### Poor Model Quality
- Increase training epochs
- Adjust learning rate
- Add more diverse training data

### Support
For issues or questions, check:
1. Configuration files are correctly formatted
2. All required packages are installed
3. GPU memory is sufficient
4. Dataset files are properly generated

## 📝 Citation

If you use this dataset or methodology, please cite:

```
EV Charging QA Dataset for Llama 3 Fine-tuning
Generated from multiple authoritative sources including:
- IEA Global EV Outlook
- DOE EV Data Collection
- Academic research papers
- Technical documentation
``` 