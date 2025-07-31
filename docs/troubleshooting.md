# Troubleshooting Guide

This guide helps you resolve common issues encountered while using the EV Charging LLM Pipeline.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Collection Problems](#data-collection-problems)
- [Training Issues](#training-issues)
- [Memory and Performance Problems](#memory-and-performance-problems)
- [Model Loading and Inference Issues](#model-loading-and-inference-issues)
- [Configuration Problems](#configuration-problems)
- [Network and API Issues](#network-and-api-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Getting Help](#getting-help)

## Installation Issues

### Python Version Incompatibility

**Problem**: `ImportError` or version-related errors during installation.

**Symptoms**:
```
ERROR: Package 'transformers' requires a different Python version
SyntaxError: f-strings require Python 3.6+
```

**Solution**:
```bash
# Check Python version
python --version
# Should be 3.8 or higher

# If using conda
conda create -n ev-charging-llm python=3.10
conda activate ev-charging-llm

# If using pyenv
pyenv install 3.10.12
pyenv local 3.10.12
```

### CUDA Installation Problems

**Problem**: PyTorch not detecting GPU or CUDA version mismatch.

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Diagnosis**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Solution**:
```bash
# Install correct PyTorch version for your CUDA
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (not recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Dependency Conflicts

**Problem**: Package version conflicts during installation.

**Symptoms**:
```
ERROR: pip's dependency resolver does not currently consider all the packages
ERROR: Could not install packages due to an EnvironmentError
```

**Solution**:
```bash
# Use fresh virtual environment
conda create -n ev-charging-llm-clean python=3.10
conda activate ev-charging-llm-clean

# Install dependencies in order
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

# If conflicts persist, use conda for conflicting packages
conda install transformers datasets -c huggingface -c conda-forge
```

## Data Collection Problems

### Network Timeout Errors

**Problem**: Data collection fails with timeout errors.

**Symptoms**:
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='...', port=443): 
Max retries exceeded with url: ... (Caused by ConnectTimeoutError)
```

**Solution**:
```python
# Increase timeout in collect_data.py
DEFAULT_TIMEOUT = 60  # Increase from 30 to 60 seconds

# Add retry mechanism
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
```

### PDF Download Failures

**Problem**: PDF downloads fail or return corrupted files.

**Symptoms**:
```
ERROR: Failed to download PDF from https://example.com/report.pdf: 403 Client Error
pypdf.errors.PdfReadError: Could not find xref table at specified location
```

**Solution**:
```python
# Add user agent to requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)

# Verify PDF content
def verify_pdf_content(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            if content.startswith(b'%PDF'):
                return True
    except:
        pass
    return False

# Alternative PDF processing
try:
    reader = pypdf.PdfReader(pdf_path)
except pypdf.errors.PdfReadError:
    # Try alternative PDF library
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
```

### Web Scraping Blocks

**Problem**: Websites blocking scraping attempts.

**Symptoms**:
```
HTTP 403 Forbidden
HTTP 429 Too Many Requests
Captcha challenges
```

**Solution**:
```python
# Implement respectful scraping
import time
import random

# Add delays between requests
time.sleep(random.uniform(1, 3))

# Rotate user agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
]

headers = {'User-Agent': random.choice(user_agents)}

# Respect robots.txt
from urllib.robotparser import RobotFileParser
def can_fetch(url):
    rp = RobotFileParser()
    rp.set_url(url + '/robots.txt')
    rp.read()
    return rp.can_fetch('*', url)
```

### Search API Rate Limits

**Problem**: DuckDuckGo or Google search APIs returning rate limit errors.

**Symptoms**:
```
ERROR: Rate limit exceeded
ERROR: Too many requests
```

**Solution**:
```python
# Implement exponential backoff
import time
import random

def search_with_backoff(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return search_function(query)
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")

# Use alternative search engines
search_engines = ['duckduckgo', 'bing', 'yandex']
for engine in search_engines:
    try:
        results = search_with_engine(query, engine)
        break
    except Exception as e:
        logger.warning(f"Search failed with {engine}: {e}")
```

## Training Issues

### Out of Memory (OOM) Errors

**Problem**: GPU runs out of memory during training.

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Immediate Solutions**:
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Kill other GPU processes
nvidia-smi
kill -9 <process_id>
```

**Long-term Solutions**:
```yaml
# Reduce batch size in lora_config.yaml
training_args:
  per_device_train_batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size

# Enable gradient checkpointing
hardware:
  use_gradient_checkpointing: true

# Use QLoRA instead of LoRA
qlora_config:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
```

### Slow Training Performance

**Problem**: Training is extremely slow.

**Symptoms**:
- Very low tokens/second
- Each epoch takes hours
- GPU utilization is low

**Diagnosis**:
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Check CPU bottlenecks
htop

# Monitor memory usage
watch -n 1 free -h
```

**Solutions**:
```yaml
# Optimize batch size
training_args:
  per_device_train_batch_size: 4  # Increase if memory allows
  dataloader_num_workers: 4       # Increase for faster data loading
  group_by_length: true           # Group similar length sequences

# Enable mixed precision
training_args:
  bf16: true  # Or fp16: true for older GPUs

# Reduce sequence length if possible
max_seq_length: 1024  # Reduce from 2048 if appropriate
```

### Training Divergence

**Problem**: Loss increases or becomes NaN during training.

**Symptoms**:
```
Step 100: loss = 2.5
Step 200: loss = 5.8
Step 300: loss = nan
```

**Solutions**:
```yaml
# Reduce learning rate
training_args:
  learning_rate: 1e-4  # Reduce from 2e-4

# Add gradient clipping
training_args:
  max_grad_norm: 1.0

# Increase warmup steps
training_args:
  warmup_steps: 200  # Increase from 100

# Check data quality
# Ensure no corrupted or extremely long sequences
```

### Checkpoint Loading Failures

**Problem**: Cannot load saved checkpoints.

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoint-1000'
RuntimeError: Error(s) in loading state_dict
```

**Solutions**:
```bash
# Verify checkpoint directory structure
ls -la models/llama3-7b-ev-charging-lora/
# Should contain: adapter_config.json, adapter_model.bin, training_args.bin

# Check for incomplete saves
du -h models/llama3-7b-ev-charging-lora/*
# Files should be substantial in size

# Manual checkpoint recovery
python -c "
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained('models/llama3-7b-ev-charging-lora')
print('Config loaded successfully')
"
```

## Memory and Performance Problems

### High CPU Usage

**Problem**: CPU usage at 100% slowing down training.

**Symptoms**:
- System becomes unresponsive
- Training speed decreases significantly
- High load average

**Solutions**:
```yaml
# Reduce data loading workers
training_args:
  dataloader_num_workers: 2  # Reduce from 4

# Limit CPU parallelism
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use process pinning
taskset -c 0-7 python train_llama3_lora.py
```

### Disk Space Issues

**Problem**: Running out of disk space during training.

**Symptoms**:
```
OSError: [Errno 28] No space left on device
```

**Prevention**:
```bash
# Monitor disk usage
df -h
du -sh data/ models/ logs/

# Clean unnecessary files
rm -rf data/pdfs/*.tmp
rm -rf models/*/checkpoint-*/
find . -name "*.log" -mtime +7 -delete

# Use symbolic links for large datasets
ln -s /large_storage/data ./data
```

### Memory Leaks

**Problem**: Memory usage continuously increases during training.

**Symptoms**:
- System memory gradually fills up
- Eventually leads to OOM killer
- Swap usage increases

**Solutions**:
```python
# Clear cache periodically
import gc
import torch

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Add to training loop
if step % 100 == 0:
    clear_memory()

# Monitor memory usage
import psutil
def log_memory_usage():
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent}%")
```

## Model Loading and Inference Issues

### Model Architecture Mismatches

**Problem**: Errors when loading fine-tuned models.

**Symptoms**:
```
RuntimeError: size mismatch for transformer.h.0.attn.q_proj.weight
KeyError: 'some_key' not found in checkpoint
```

**Solutions**:
```python
# Verify model compatibility
from transformers import AutoConfig

base_config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
adapter_config = PeftConfig.from_pretrained("models/llama3-7b-ev-charging-lora")

print(f"Base model: {base_config.model_type}")
print(f"Adapter target modules: {adapter_config.target_modules}")

# Load with strict=False for debugging
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-7B-Instruct",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, "models/llama3-7b-ev-charging-lora")
```

### Tokenizer Issues

**Problem**: Text generation produces garbled or incorrect output.

**Symptoms**:
- Random characters in output
- Incomplete responses
- Encoding errors

**Solutions**:
```python
# Verify tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")

# Test tokenization
test_text = "What are EV charging connectors?"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)
print(f"Original: {test_text}")
print(f"Decoded: {decoded}")

# Fix padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### Generation Quality Issues

**Problem**: Model generates poor quality or irrelevant responses.

**Symptoms**:
- Repetitive text
- Off-topic responses
- Very short or very long responses

**Solutions**:
```python
# Adjust generation parameters
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,      # Lower for more focused responses
    "top_p": 0.9,           # Nucleus sampling
    "top_k": 50,            # Top-k sampling
    "repetition_penalty": 1.1,  # Reduce repetition
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

# Use proper prompt format
def format_prompt(question):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert on electric vehicle charging technology.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Configuration Problems

### YAML Syntax Errors

**Problem**: Configuration files have syntax errors.

**Symptoms**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
yaml.parser.ParserError: expected <block end>, but found '<scalar>'
```

**Solutions**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Common fixes:
# 1. Check indentation (use spaces, not tabs)
# 2. Quote strings with special characters
# 3. Use proper list format

# Example correct format:
```

```yaml
# Correct YAML format
training_args:
  learning_rate: 2e-4
  target_modules:
    - "q_proj"
    - "k_proj"
  special_chars: "quotes needed: for colons"
```

### Path Configuration Issues

**Problem**: File paths in configuration don't work.

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train_lora.json'
```

**Solutions**:
```python
# Use absolute paths for clarity
import os
from pathlib import Path

config_dir = Path(__file__).parent
data_dir = config_dir / "data"

config = {
    "train_dataset": str(data_dir / "train_lora.json"),
    "eval_dataset": str(data_dir / "eval_lora.json")
}

# Verify paths exist
for path in [config["train_dataset"], config["eval_dataset"]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
```

## Network and API Issues

### Proxy and Firewall Problems

**Problem**: Network requests fail due to corporate firewalls or proxies.

**Symptoms**:
```
requests.exceptions.ProxyError: HTTPSConnectionPool
requests.exceptions.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions**:
```python
# Configure proxy settings
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'

# Or in requests
proxies = {
    'http': 'http://proxy.company.com:8080',
    'https': 'http://proxy.company.com:8080'
}
response = requests.get(url, proxies=proxies)

# For SSL issues (use carefully)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
response = requests.get(url, verify=False)  # Only for testing
```

### DNS Resolution Issues

**Problem**: Cannot resolve hostnames for data sources.

**Symptoms**:
```
requests.exceptions.ConnectionError: [Errno 11001] getaddrinfo failed
```

**Solutions**:
```bash
# Test DNS resolution
nslookup api.openchargemap.io
ping api.openchargemap.io

# Use alternative DNS servers
# Windows: Change DNS in network settings
# Linux: Edit /etc/resolv.conf
nameserver 8.8.8.8
nameserver 8.8.4.4

# Python DNS override
import socket
socket.getaddrinfo('api.openchargemap.io', 443)
```

## Platform-Specific Issues

### Windows-Specific Problems

**Problem**: Path separator issues on Windows.

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/pdfs\file.pdf'
```

**Solutions**:
```python
# Use pathlib for cross-platform paths
from pathlib import Path

pdf_dir = Path("data") / "pdfs"
pdf_file = pdf_dir / "document.pdf"

# Or use os.path.join
import os
pdf_path = os.path.join("data", "pdfs", "document.pdf")
```

**Problem**: Long path names on Windows.

**Solutions**:
```bash
# Enable long paths in Windows 10/11
# Run as administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use UNC paths
\\?\C:\very\long\path\to\your\data
```

### macOS-Specific Problems

**Problem**: Permission issues with Apple Silicon Macs.

**Solutions**:
```bash
# Install Rosetta 2 for compatibility
softwareupdate --install-rosetta

# Use conda-forge for arm64 packages
conda install pytorch torchvision torchaudio -c conda-forge

# Set architecture explicitly
arch -arm64 pip install transformers
```

### Linux-Specific Problems

**Problem**: CUDA driver issues on Linux.

**Solutions**:
```bash
# Check driver installation
nvidia-smi
sudo apt update && sudo apt install nvidia-driver-535

# Verify CUDA installation
ls /usr/local/cuda/bin/
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
python --version
pip list | grep -E "(torch|transformers|peft|datasets)"

# GPU information
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Memory information
free -h
df -h

# Process information
ps aux | grep python
```

### Log Collection

```bash
# Enable debug logging
export PYTHONPATH="."
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Redirect output to file
python train_llama3_lora.py 2>&1 | tee training.log

# Collect relevant logs
grep -E "(ERROR|CRITICAL|Exception)" training.log > errors.log
```

### Community Support

1. **GitHub Issues**: [Create an issue](https://github.com/yourusername/ev-charging-llm-pipeline/issues)
2. **GitHub Discussions**: [Join discussions](https://github.com/yourusername/ev-charging-llm-pipeline/discussions)
3. **Documentation**: Check [API Reference](api_reference.md) and [Training Guide](training_guide.md)

### Issue Template

When reporting bugs, use this template:

```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: [e.g., Ubuntu 20.04, Windows 11, macOS 13]
- Python version: [e.g., 3.10.12]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 4090]
- Package versions: [output of pip list]

## Steps to Reproduce
1. Step one
2. Step two
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
Full error traceback here
```

## Additional Context
Any other relevant information
```

### Performance Issues

For performance-related issues:

1. **Provide benchmarks**: Include training speed, memory usage
2. **Hardware specs**: Detailed GPU, CPU, RAM information
3. **Configuration**: Share your `lora_config.yaml`
4. **Comparison**: How it compares to expected performance

### Training Quality Issues

For model quality issues:

1. **Example outputs**: Share generated responses
2. **Training logs**: Include loss curves and metrics
3. **Data quality**: Information about your training data
4. **Hyperparameters**: Your training configuration

---

Remember: Most issues have been encountered before. Check existing issues and discussions before creating new ones. The community is here to help! 🚗⚡ 