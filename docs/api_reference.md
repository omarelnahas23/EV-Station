# API Reference

This document provides detailed information about the EV Charging LLM Pipeline API, including all modules, classes, and functions.

## Table of Contents

- [Data Collection API](#data-collection-api)
- [Data Processing API](#data-processing-api)
- [Model Training API](#model-training-api)
- [Model Inference API](#model-inference-api)
- [Configuration API](#configuration-api)

## Data Collection API

### Module: `data_collection.collect_data`

Main module for collecting EV charging data from various sources.

#### Functions

##### `fetch_ocm_data(api_url, api_key=None, country_code=None, max_results=100)`

Fetch EV charging station data from the OpenChargeMap API.

**Parameters:**
- `api_url` (str): OpenChargeMap API endpoint URL
- `api_key` (str, optional): API key for authenticated requests
- `country_code` (str, optional): ISO country code filter (e.g., "US", "GB")
- `max_results` (int): Maximum number of results to return (default: 100)

**Returns:**
- `List[Dict[str, Any]]`: List of charging station data dictionaries

**Example:**
```python
from data_collection.collect_data import fetch_ocm_data

api_url = "https://api.openchargemap.io/v3/poi"
stations = fetch_ocm_data(api_url, country_code="US", max_results=50)
```

##### `download_pdf(url, output_dir)`

Download a PDF document from a URL and save it locally.

**Parameters:**
- `url` (str): URL of the PDF document to download
- `output_dir` (Path): Directory to save the downloaded PDF

**Returns:**
- `Path`: Path to the downloaded PDF file

**Raises:**
- `requests.exceptions.RequestException`: If download fails
- `OSError`: If file cannot be saved

**Example:**
```python
from pathlib import Path
from data_collection.collect_data import download_pdf

pdf_url = "https://example.com/ev_report.pdf"
output_dir = Path("data/pdfs")
pdf_path = download_pdf(pdf_url, output_dir)
```

##### `extract_text_from_pdf(pdf_path)`

Extract text content from a PDF file.

**Parameters:**
- `pdf_path` (Path): Path to the PDF file

**Returns:**
- `str`: Extracted text content from all pages

**Raises:**
- `pypdf.errors.PdfReadError`: If PDF cannot be read
- `FileNotFoundError`: If PDF file doesn't exist

**Example:**
```python
from pathlib import Path
from data_collection.collect_data import extract_text_from_pdf

pdf_path = Path("data/pdfs/ev_report.pdf")
text_content = extract_text_from_pdf(pdf_path)
```

##### `search_duckduckgo(query, max_results=10)`

Search DuckDuckGo for EV charging related content.

**Parameters:**
- `query` (str): Search query string
- `max_results` (int): Maximum number of results (default: 10)

**Returns:**
- `List[Dict[str, str]]`: List of search result dictionaries with 'title', 'url', 'snippet'

**Example:**
```python
from data_collection.collect_data import search_duckduckgo

results = search_duckduckgo("EV charging standards", max_results=5)
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
```

##### `scrape_web(url, source)`

Scrape web content from a given URL.

**Parameters:**
- `url` (str): URL to scrape
- `source` (str): Source identifier for metadata

**Returns:**
- `Dict[str, Any]`: Dictionary with 'text' and 'metadata' keys

**Example:**
```python
from data_collection.collect_data import scrape_web

url = "https://en.wikipedia.org/wiki/Electric_vehicle_charging_station"
content = scrape_web(url, "Wikipedia - EV Charging")
```

##### `collect_data(config)`

Main data collection pipeline function.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary loaded from config.yaml

**Returns:**
- `List[Dict[str, Any]]`: Collected data items from all sources

**Example:**
```python
import yaml
from data_collection.collect_data import collect_data

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
data = collect_data(config)
```

## Data Processing API

### Module: `data_processing.process_data`

Module for cleaning, filtering, and preprocessing collected data.

#### Functions

##### `clean_text(text)`

Clean and normalize text data.

**Parameters:**
- `text` (str): Raw text to clean

**Returns:**
- `str`: Cleaned and normalized text

**Example:**
```python
from data_processing.process_data import clean_text

raw_text = "  This is   messy\t\ttext\n\n\nwith   extra   spaces  "
clean_text_result = clean_text(raw_text)
# Result: "This is messy text with extra spaces"
```

##### `filter_quality(data_item, min_length=50, max_length=5000)`

Filter data based on quality criteria.

**Parameters:**
- `data_item` (Dict[str, Any]): Data item with 'text' key
- `min_length` (int): Minimum text length (default: 50)
- `max_length` (int): Maximum text length (default: 5000)

**Returns:**
- `bool`: True if data meets quality criteria

**Example:**
```python
from data_processing.process_data import filter_quality

data_item = {"text": "This is a good quality text with meaningful content."}
is_quality = filter_quality(data_item, min_length=20)
```

##### `chunk_text(text, chunk_size=512, overlap=50)`

Split long text into smaller chunks with overlap.

**Parameters:**
- `text` (str): Text to chunk
- `chunk_size` (int): Target chunk size in words (default: 512)
- `overlap` (int): Overlap between chunks in words (default: 50)

**Returns:**
- `List[str]`: List of text chunks

**Example:**
```python
from data_processing.process_data import chunk_text

long_text = "This is a very long text that needs to be chunked..."
chunks = chunk_text(long_text, chunk_size=100, overlap=20)
```

### Module: `data_processing.generate_qa_dataset`

Module for generating question-answer pairs from processed data.

#### Classes

##### `EVChargingQAGenerator`

Main class for generating QA pairs from EV charging data.

**Methods:**

###### `__init__()`

Initialize the QA generator with predefined question templates and topic keywords.

###### `categorize_content(text)`

Categorize content based on keywords to match with appropriate questions.

**Parameters:**
- `text` (str): Text content to categorize

**Returns:**
- `str`: Topic category ('connector_types', 'charging_levels', etc.)

###### `generate_qa_pairs(processed_data)`

Generate QA pairs from processed data.

**Parameters:**
- `processed_data` (List[Dict[str, Any]]): List of processed data items

**Returns:**
- `List[Dict[str, str]]`: List of QA pair dictionaries

**Example:**
```python
from data_processing.generate_qa_dataset import EVChargingQAGenerator

generator = EVChargingQAGenerator()
qa_pairs = generator.generate_qa_pairs(processed_data)
```

###### `create_llama3_format(question, answer, system_prompt=None)`

Format QA pair for Llama 3 training.

**Parameters:**
- `question` (str): Question text
- `answer` (str): Answer text
- `system_prompt` (str, optional): System prompt for the model

**Returns:**
- `Dict[str, str]`: Formatted QA pair with Llama 3 chat format

#### Functions

##### `generate_qa_dataset(config)`

Main function to generate QA dataset for Llama 3 fine-tuning.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `List[Dict[str, str]]`: Generated QA pairs

**Example:**
```python
import yaml
from data_processing.generate_qa_dataset import generate_qa_dataset

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
qa_pairs = generate_qa_dataset(config)
```

## Model Training API

### Module: `data_processing.train_llama3_lora`

Module for training Llama 3 models with LoRA/QLoRA.

#### Functions

##### `load_config(config_path="lora_config.yaml")`

Load training configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

##### `setup_model_and_tokenizer(config)`

Setup model and tokenizer with optional quantization.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Tuple[AutoModelForCausalLM, AutoTokenizer]`: Model and tokenizer

##### `setup_lora(model, config)`

Setup LoRA configuration for the model.

**Parameters:**
- `model` (AutoModelForCausalLM): Base model
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `PeftModel`: Model with LoRA adapters

##### `main()`

Main training function that orchestrates the entire training process.

**Example:**
```python
from data_processing.train_llama3_lora import main

# Start training with default configuration
main()
```

## Model Inference API

### Module: `data_processing.inference_llama3`

Module for running inference on fine-tuned models.

#### Classes

##### `EVChargingChatbot`

Chatbot interface for the fine-tuned EV charging model.

**Methods:**

###### `__init__(model_path, base_model="meta-llama/Meta-Llama-3-7B-Instruct")`

Initialize the chatbot with a fine-tuned model.

**Parameters:**
- `model_path` (str): Path to the fine-tuned LoRA model
- `base_model` (str): Base model identifier

###### `generate_response(question, max_new_tokens=512, temperature=0.7)`

Generate a response to a question.

**Parameters:**
- `question` (str): User question
- `max_new_tokens` (int): Maximum tokens to generate (default: 512)
- `temperature` (float): Sampling temperature (default: 0.7)

**Returns:**
- `str`: Generated response

**Example:**
```python
from data_processing.inference_llama3 import EVChargingChatbot

chatbot = EVChargingChatbot("models/llama3-7b-ev-charging-lora")
response = chatbot.generate_response("What are the types of EV connectors?")
```

###### `interactive_chat()`

Start an interactive chat session.

**Example:**
```python
chatbot = EVChargingChatbot("models/llama3-7b-ev-charging-lora")
chatbot.interactive_chat()  # Starts interactive session
```

#### Functions

##### `test_model(model_path)`

Test the model with predefined benchmark questions.

**Parameters:**
- `model_path` (str): Path to the fine-tuned model

**Example:**
```python
from data_processing.inference_llama3 import test_model

test_model("models/llama3-7b-ev-charging-lora")
```

## Configuration API

### Configuration Files

#### `config.yaml`

Main configuration file for the entire pipeline.

**Structure:**
```yaml
domain: 'electric vehicle charging stations'
use_case: 'QA'
base_model: 'meta-llama/Meta-Llama-3-7B-Instruct'
data_dir: 'data'
training_data_file: 'data/train_dataset.json'
eval_data_file: 'data/eval_dataset.json'
fine_tuned_model_dir: 'models/fine_tuned'
enable_web_search: true
enable_pdf_download: true
enable_api_collection: true
```

#### `data_processing/lora_config.yaml`

LoRA/QLoRA training configuration.

**Key Sections:**
- `lora_config`: LoRA-specific parameters
- `qlora_config`: QLoRA quantization settings
- `training_args`: Training hyperparameters
- `hardware`: Hardware optimization settings

## Error Handling

### Common Exceptions

- `requests.exceptions.RequestException`: Network-related errors during data collection
- `pypdf.errors.PdfReadError`: PDF processing errors
- `FileNotFoundError`: Missing files or directories
- `json.JSONDecodeError`: JSON parsing errors
- `ValueError`: Invalid parameter values
- `RuntimeError`: CUDA or model loading errors

### Error Recovery

The pipeline includes robust error handling:
- Automatic retries for network requests
- Graceful degradation when data sources fail
- Comprehensive logging for debugging
- Validation of inputs and outputs

## Performance Considerations

### Memory Management

- Use QLoRA for GPU memory constraints (8GB+)
- Enable gradient checkpointing for large models
- Process data in chunks for large datasets

### Speed Optimization

- Parallel processing for data collection
- Caching for repeated operations
- Efficient tokenization strategies

### Best Practices

- Monitor GPU memory usage during training
- Use appropriate batch sizes for your hardware
- Validate data quality before training
- Regular checkpointing during long training runs

## Version Information

- **API Version**: 1.0.0
- **Compatible Models**: Llama 3 7B, Llama 2 7B/13B
- **Python Requirements**: 3.8+
- **CUDA Requirements**: 11.8+ (for GPU training)

For more detailed examples and tutorials, see the [Training Guide](training_guide.md). 