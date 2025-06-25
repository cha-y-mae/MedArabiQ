# ArabicLLMBench

This repository benchmarks Arabic large language models (LLMs) for various tasks such as question-answering (QA), summarization, dialogue simulation, and clinical note generation.

## Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Evaluation](#evaluation)
6. [Contributing](#contributing)

## Overview
ArabicLLMBench is a framework for evaluating and benchmarking language models on Arabic datasets. It supports multiple model types, including OpenAI and Hugging Face

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ArabicLLMBench.git
   cd ArabicLLMBench
   
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   
4. Set API keys
   ```bash 
    export OPENAI_API_KEY="your_openai_api_key_here"
    export HF_TOKEN="your_huggingface_api_key_here"
    export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
    export ANTHROPIC_API_KEY=""
    export GEMINI_API_KEY=""

5. Modify/create config file
    Config files are stored under folder /configs and they define model type, model name, dataset paths etc.
     Make sure to specify your cache_dir in the config!

7. Run experiment
   To run an experiment, make sure you execute the script from the root directory of the repository.
   Change 2nd argument to the path of the desired config file. Example below:
   ```bash
   python scripts/run_experiment.py configs/fib_closed_falcon.yaml 


   
   
