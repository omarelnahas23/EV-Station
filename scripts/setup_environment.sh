#!/bin/bash

# EV Charging LLM Pipeline - Environment Setup Script
# This script automates the setup process for the EV Charging LLM Pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_header() {
    echo
    print_color $BLUE "=================================================="
    print_color $BLUE "$1"
    print_color $BLUE "=================================================="
    echo
}

print_success() {
    print_color $GREEN "✅ $1"
}

print_warning() {
    print_color $YELLOW "⚠️  $1"
}

print_error() {
    print_color $RED "❌ $1"
}

print_info() {
    print_color $BLUE "ℹ️  $1"
}

# Check if running on supported OS
check_os() {
    print_header "Checking Operating System"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "Detected macOS system"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Detected Windows system (WSL/Git Bash)"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check for required commands
check_dependencies() {
    print_header "Checking Dependencies"
    
    REQUIRED_COMMANDS=("python3" "pip" "git")
    
    for cmd in "${REQUIRED_COMMANDS[@]}"; do
        if command -v $cmd &> /dev/null; then
            print_success "$cmd is installed"
        else
            print_error "$cmd is not installed. Please install it first."
            exit 1
        fi
    done
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python $PYTHON_VERSION is compatible"
    else
        print_error "Python $PYTHON_VERSION is too old. Python 3.8+ required."
        exit 1
    fi
}

# Check for CUDA
check_cuda() {
    print_header "Checking CUDA"
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        if [ ! -z "$CUDA_VERSION" ]; then
            print_success "CUDA $CUDA_VERSION detected"
            CUDA_AVAILABLE=true
        else
            print_warning "nvidia-smi found but CUDA version unclear"
            CUDA_AVAILABLE=false
        fi
    else
        print_warning "CUDA not detected. Training will be slower on CPU."
        CUDA_AVAILABLE=false
    fi
}

# Setup virtual environment
setup_venv() {
    print_header "Setting up Virtual Environment"
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        print_info "Using conda for environment management"
        
        # Create conda environment
        ENV_NAME="ev-charging-llm"
        if conda env list | grep -q "^$ENV_NAME "; then
            print_warning "Environment $ENV_NAME already exists. Removing..."
            conda env remove -n $ENV_NAME -y
        fi
        
        print_info "Creating conda environment: $ENV_NAME"
        conda create -n $ENV_NAME python=3.10 -y
        
        print_success "Conda environment created"
        print_info "To activate: conda activate $ENV_NAME"
        
        # Activate environment for the rest of the script
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
        
        VENV_TYPE="conda"
        
    else
        print_info "Using venv for environment management"
        
        # Create virtual environment
        VENV_DIR="venv"
        if [ -d "$VENV_DIR" ]; then
            print_warning "Virtual environment already exists. Removing..."
            rm -rf $VENV_DIR
        fi
        
        print_info "Creating virtual environment: $VENV_DIR"
        python3 -m venv $VENV_DIR
        
        # Activate virtual environment
        if [[ "$OS" == "windows" ]]; then
            source $VENV_DIR/Scripts/activate
        else
            source $VENV_DIR/bin/activate
        fi
        
        print_success "Virtual environment created and activated"
        print_info "To activate later: source $VENV_DIR/bin/activate"
        
        VENV_TYPE="venv"
    fi
}

# Install PyTorch with appropriate CUDA support
install_pytorch() {
    print_header "Installing PyTorch"
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        # Determine CUDA version for PyTorch
        if [[ "$CUDA_VERSION" == "11.8"* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        elif [[ "$CUDA_VERSION" == "12.1"* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        else
            print_warning "Unsupported CUDA version $CUDA_VERSION, using default"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        fi
        
        print_info "Installing PyTorch with CUDA $CUDA_VERSION support"
        pip install torch torchvision torchaudio --index-url $TORCH_INDEX
    else
        print_info "Installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Verify PyTorch installation
    python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    fi
    
    print_success "PyTorch installed successfully"
}

# Install project dependencies
install_dependencies() {
    print_header "Installing Project Dependencies"
    
    print_info "Installing core dependencies..."
    pip install -r requirements.txt
    
    print_info "Installing training dependencies..."
    pip install transformers[torch] peft bitsandbytes datasets accelerate
    
    print_success "Dependencies installed successfully"
}

# Setup development environment (optional)
setup_dev_environment() {
    print_header "Setting up Development Environment (Optional)"
    
    read -p "Do you want to install development dependencies? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
        
        # Setup pre-commit hooks
        if command -v pre-commit &> /dev/null; then
            print_info "Setting up pre-commit hooks..."
            pre-commit install
            print_success "Pre-commit hooks installed"
        fi
        
        print_success "Development environment setup complete"
    else
        print_info "Skipping development dependencies"
    fi
}

# Setup Weights & Biases (optional)
setup_wandb() {
    print_header "Setting up Weights & Biases (Optional)"
    
    read -p "Do you want to setup Weights & Biases for experiment tracking? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v wandb &> /dev/null; then
            print_info "Weights & Biases is already installed"
        else
            print_info "Installing Weights & Biases..."
            pip install wandb
        fi
        
        print_info "Please login to Weights & Biases:"
        wandb login
        
        print_success "Weights & Biases setup complete"
    else
        print_info "Skipping Weights & Biases setup"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating Project Directories"
    
    DIRECTORIES=(
        "data"
        "data/pdfs"
        "models"
        "logs"
        "data_collection/data"
        "data_processing/data"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    print_info "Testing imports..."
    
    # Test core imports
    python3 -c "
import torch
import transformers
import datasets
import peft
import yaml
import requests
import bs4

print('✅ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
    
    print_success "Installation verification complete"
}

# Create activation script
create_activation_script() {
    print_header "Creating Activation Script"
    
    SCRIPT_NAME="activate_env.sh"
    
    cat > $SCRIPT_NAME << EOF
#!/bin/bash
# Activation script for EV Charging LLM Pipeline

echo "🚗⚡ Activating EV Charging LLM Pipeline Environment"

EOF
    
    if [ "$VENV_TYPE" = "conda" ]; then
        cat >> $SCRIPT_NAME << EOF
# Activate conda environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ev-charging-llm

echo "✅ Conda environment 'ev-charging-llm' activated"
EOF
    else
        cat >> $SCRIPT_NAME << EOF
# Activate virtual environment
if [[ "\$OSTYPE" == "msys" ]] || [[ "\$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"
EOF
    fi
    
    cat >> $SCRIPT_NAME << EOF

# Set environment variables
export PYTHONPATH="\$PWD:\$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

echo "✅ Environment variables set"
echo "Ready to use EV Charging LLM Pipeline!"
echo
echo "Quick commands:"
echo "  Data collection:    cd data_collection && python collect_data.py"
echo "  Data processing:    cd data_processing && python process_data.py"
echo "  Generate QA pairs:  cd data_processing && python generate_qa_dataset.py"
echo "  Train model:        cd data_processing && python train_llama3_lora.py"
echo
EOF
    
    chmod +x $SCRIPT_NAME
    print_success "Created activation script: $SCRIPT_NAME"
}

# Main setup function
main() {
    print_header "EV Charging LLM Pipeline - Environment Setup"
    print_info "This script will set up your environment for the EV Charging LLM Pipeline"
    echo
    
    # Check if we're in the right directory
    if [ ! -f "config.yaml" ] || [ ! -d "data_collection" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    check_os
    check_dependencies
    check_cuda
    setup_venv
    install_pytorch
    install_dependencies
    setup_dev_environment
    setup_wandb
    create_directories
    verify_installation
    create_activation_script
    
    print_header "Setup Complete!"
    print_success "EV Charging LLM Pipeline environment is ready to use!"
    echo
    print_info "Next steps:"
    print_info "1. Run: source activate_env.sh"
    print_info "2. Start data collection: cd data_collection && python collect_data.py"
    print_info "3. Process data: cd data_processing && python process_data.py"
    print_info "4. Generate QA dataset: python generate_qa_dataset.py"
    print_info "5. Train model: python train_llama3_lora.py"
    echo
    print_info "For help, see docs/training_guide.md or docs/troubleshooting.md"
    echo
    print_color $GREEN "Happy training! 🚗⚡"
}

# Run main function
main "$@" 