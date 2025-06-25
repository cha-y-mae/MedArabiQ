from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import os
from huggingface_hub import login
import re
import torch

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/scratch/ca2627/huggingface"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN environment variable.")

class medgemma:
    def __init__(self, model_name, cache_dir=None):
        """
        Initialize the Hugging Face handler with a configurable cache directory.

        Args:
            model_name (str): Name of the Hugging Face model.
            cache_dir (str): Directory to cache models and tokenizers.
        """        
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Clear any existing GPU memory before loading
        torch.cuda.empty_cache()
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} initial memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f}GB")

        print("Starting model loading...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Force accelerate to use GPUs
        from accelerate import infer_auto_device_map, init_empty_weights
        
        # First, get the model config to calculate device map
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Create device map manually
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        device_map = infer_auto_device_map(
            model, 
            max_memory={0: "30GB", 1: "30GB"},
            dtype=torch.float16
        )
        
        print("Calculated device map:")
        for k, v in device_map.items():
            print(f"  {k}: {v}")
        
        # Now load with the explicit device map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/scratch/ca2627/huggingface",
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("Model loading completed!")

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.tokenizer.padding_side = "left"

        print("Model loaded successfully!")
        
        # Print GPU distribution info
        print(f"Available GPUs: {torch.cuda.device_count()}")
        if hasattr(self.model, 'hf_device_map'):
            print("Model device distribution:")
            for layer, device in self.model.hf_device_map.items():
                print(f"  {layer}: {device}")
        
        # Check memory usage on both GPUs
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Dynamically set the pad_token if it is missing
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize the text-generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=1,
            trust_remote_code=True
        )
        print("Using the text-generation pipeline.")

    def cpu_fallback_generation(self, messages, max_tokens):
        """
        Simple fallback that returns a placeholder instead of loading CPU model
        """
        print("CUDA error detected - returning placeholder response...")
        return "Unable to generate response due to CUDA compatibility issues with this text."

    def prompt(self, input_text, instruction, task_type, max_tokens=256):
        """
        Processes a question based on the task type and queries the model using the text-generation pipeline.
        """

        try:
            # Test tokenization first
            test_tokens = self.tokenizer.encode(input_text, return_tensors="pt")
            print(f"Input tokens shape: {test_tokens.shape}")
            print(f"Max token ID: {test_tokens.max().item()}")
            print(f"Vocab size: {self.tokenizer.vocab_size}")
            
            # Check if any token IDs exceed vocab size
            if test_tokens.max().item() >= self.tokenizer.vocab_size:
                print("WARNING: Token ID exceeds vocabulary size!")
                return "Tokenization error"
            
        except Exception as e:
            print(f"Tokenization error: {e}")
            return "Tokenization failed"
    
        if task_type in ["fib_open", "aramed"]:
            # For open-ended fill-in-the-blank questions, extract the question without options
            question = input_text.strip()
    
            # Use chat format for generation
            messages = [
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
    
            try:
                # Clear memory before each generation
                torch.cuda.empty_cache()
                
                print("Starting GPU generation...")
                
                # Use chat format instead of raw prompt
                response = self.pipeline(
                    messages, 
                    max_new_tokens=min(max_tokens, 150),
                    num_return_sequences=1, 
                    temperature=0.2, 
                    top_p=0.9, 
                    do_sample=True
                )

                print("GPU generation completed!")
                
                # Extract from chat response format
                generated_text = response[0]["generated_text"][-1]["content"].strip()
                
                # Clear memory after generation
                torch.cuda.empty_cache()
                
                return generated_text

            except RuntimeError as e:
                if "same device" in str(e) and "cuda" in str(e):
                    print(f"Multi-GPU device mismatch detected, trying single GPU fallback...")
                    torch.cuda.empty_cache()
                    try:
                        # For multi-GPU models, try moving to single GPU temporarily
                        # Create a simple prompt and use direct generation
                        system_msg = messages[0]['content']
                        user_msg = messages[1]['content']
                        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                        
                        # Tokenize and move to the model's first device
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        model_device = next(self.model.parameters()).device
                        inputs = {k: v.to(model_device) for k, v in inputs.items()}
                        
                        # Direct generation bypassing pipeline
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=min(max_tokens, 100),
                                temperature=0.2,
                                do_sample=True,
                                pad_token_id=self.tokenizer.pad_token_id
                            )
                        
                        # Decode response
                        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        generated_part = response.replace(prompt, "").strip()
                        
                        torch.cuda.empty_cache()
                        return generated_part
                        
                    except Exception as retry_e:
                        print(f"Direct generation also failed: {str(retry_e)}")
                        return self.cpu_fallback_generation(messages, max_tokens)
                elif "CUDA" in str(e) and "assert" in str(e):
                    print(f"CUDA device assert error detected, trying CPU fallback...")
                    return self.cpu_fallback_generation(messages, max_tokens)
                elif "out of memory" in str(e).lower():
                    print(f"GPU out of memory, clearing cache and retrying with smaller tokens...")
                    torch.cuda.empty_cache()
                    try:
                        # Retry with much smaller token limit
                        response = self.pipeline(
                            messages, 
                            max_new_tokens=50,
                            num_return_sequences=1, 
                            temperature=0.2,
                            do_sample=False
                        )
                        generated_text = response[0]["generated_text"][-1]["content"].strip()
                        torch.cuda.empty_cache()
                        return generated_text
                    except Exception as retry_e:
                        print(f"Retry also failed: {str(retry_e)}")
                        torch.cuda.empty_cache()
                        return "Error: Unable to generate due to memory constraints"
                else:
                    print(f"Error occurred while processing the input: {input_text}")
                    print(f"Error details: {str(e)}")
                    return None
            except Exception as e:
                print(f"Error occurred while processing the input: {input_text}")
                print(f"Error details: {str(e)}")
                torch.cuda.empty_cache()
                return None

        else:  # Default to handling multiple-choice questions (for qa and fib_closed)
            options = re.findall(r"[أبجد]\.\s*[^\n]+", input_text)
    
            if not options:
                print("No options found in the input text.")
                return "Invalid input format."
    
            question = input_text.split(options[0])[0].strip()
    
            # Create chat format for multiple choice
            formatted_question = f"{question}\n\n" + "\n".join(options)
            messages = [
                {
                    "role": "system",
                    "content": f"{instruction} Answer with only the Arabic letter (أ, ب, ج, د, or ه)."
                },
                {
                    "role": "user",
                    "content": formatted_question
                }
            ]
    
            try:
                # Clear CUDA cache before generation
                torch.cuda.empty_cache()
                
                # Use chat format with conservative parameters
                response = self.pipeline(
                    messages, 
                    max_new_tokens=10,
                    num_return_sequences=1, 
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
    
                # Extract from chat response format
                generated_text = response[0]["generated_text"][-1]["content"].strip()
                
                # Look for Arabic letters directly
                arabic_letter_match = re.search(r'([أبجدهو])', generated_text)
                if arabic_letter_match:
                    final_answer = arabic_letter_match.group(1)
                    print(f"Extracted letter: {final_answer}")
                else:
                    final_answer = generated_text
                    print(f"No Arabic letter found, returning: {final_answer}")
    
                return final_answer

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory for multiple choice, clearing cache...")
                    torch.cuda.empty_cache()
                    try:
                        # Retry with minimal parameters
                        response = self.pipeline(
                            messages, 
                            max_new_tokens=3,
                            num_return_sequences=1, 
                            do_sample=False
                        )
                        generated_text = response[0]["generated_text"][-1]["content"].strip()
                        arabic_letter_match = re.search(r'([أبجدهو])', generated_text)
                        if arabic_letter_match:
                            final_answer = arabic_letter_match.group(1)
                        else:
                            final_answer = generated_text
                        torch.cuda.empty_cache()
                        return final_answer
                    except Exception as retry_e:
                        print(f"Retry also failed: {str(retry_e)}")
                        torch.cuda.empty_cache()
                        return "Error: Memory constraint"
                else:
                    print(f"Error occurred while processing the input: {input_text}")
                    print(f"Error details: {str(e)}")
                    return None
            except Exception as e:
                print(f"Error occurred while processing the input: {input_text}")
                print(f"Error details: {str(e)}")
                torch.cuda.empty_cache()
                return None