from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import os
from huggingface_hub import login
import torch

# Set environment
os.environ["HF_HOME"] = "/scratch/ca2627/huggingface"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Login
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Hugging Face token not found.")

# Load model and tokenizer with quantization
model_name = "google/medgemma-27b-text-it"

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model with 4-bit quantization across multiple GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs
    cache_dir="/scratch/ca2627/huggingface",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16
)

print("Model device map:")
if hasattr(model, 'hf_device_map'):
    for layer, device in model.hf_device_map.items():
        print(f"  {layer}: {device}")
else:
    print(f"  Model on: {next(model.parameters()).device}")

print(f"Available GPUs: {torch.cuda.device_count()}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="/scratch/ca2627/huggingface"
)

# Create pipeline with quantized model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("Pipeline created successfully!")
print("Starting inference test...")

# Test with proper chat format
messages = [
    {
        "role": "system",
        "content": "You are a helpful medical assistant. Answer with the letter (A, B, C or D) only of the correct answer. Don't generate any explanations."
    },
    {
        "role": "user",
        "content": "What is the most common cause of pneumonia in adults? Choose from: A. Bacteria B. Virus C. Fungus D. Parasite"
    }
]

print("Testing English prompt with chat format...")
print(f"Messages: {messages}")
print("Calling pipeline...")

try:
    print("Starting generation with pipeline...")
    output = pipe(messages, max_new_tokens=100, do_sample=False)
    print("Generation completed!")
    generated_response = output[0]["generated_text"][-1]["content"]
    print(f"✅ SUCCESS! Generated: '{generated_response}'")
    
except Exception as e:
    print(f"❌ Pipeline ERROR: {str(e)}")
    print("Trying direct model generation instead...")
    
    try:
        # Alternative: Direct model generation
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"Input length: {input_len} tokens")
        
        print("Starting direct generation...")
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = tokenizer.decode(generation, skip_special_tokens=True)
        print(f"✅ Direct generation SUCCESS! Generated: '{decoded}'")
        
    except Exception as e2:
        print(f"❌ Direct generation ERROR: {str(e2)}")
        import traceback
        traceback.print_exc()