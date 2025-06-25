#!/bin/bash
echo "Setting up ArabicLLMBench models..."
export HF_HOME="./cache"

# Install requirements if not already installed
pip install transformers torch huggingface_hub

# Download models (they'll cache automatically)
python -c "
from transformers import AutoTokenizer, AutoModel
from models_config import REQUIRED_MODELS

for model in REQUIRED_MODELS:
    print(f'Downloading {model}...')
    try:
        AutoTokenizer.from_pretrained(model)
        AutoModel.from_pretrained(model)
        print(f'✓ {model} downloaded')
    except Exception as e:
        print(f'✗ Failed to download {model}: {e}')
"
