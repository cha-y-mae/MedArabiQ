from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import os
from huggingface_hub import login
import re
import torch

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/scratch/ca2627/huggingface"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Login to Hugging Face
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN environment variable.")

class HuggingFaceHandler:
    def __init__(self, model_name, cache_dir=None):
        """
        Initialize the Hugging Face handler with a configurable cache directory.

        Args:
            model_name (str): Name of the Hugging Face model.
            cache_dir (str): Directory to cache models and tokenizers.
            device (str): The device to use ("cpu" or "cuda").
        """        
        self.model_name = model_name
        self.cache_dir = cache_dir

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="/scratch/ca2627/huggingface",
            device_map="auto",
            offload_folder="/scratch/ca2627/offload",
            trust_remote_code=True, #remove for models other than jais
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True,     # Reduce CPU memory usage
            max_memory={0: "35GB"}      # Limit GPU memory usage
        )

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.tokenizer.padding_side = "left"

        print(self.model.device)  # Should show 'cuda:0' or similar

        # Dynamically set the pad_token if it is missing
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize the text-generation pipeline (since Falcon is a generative model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=1,  # Reduce batch size
            trust_remote_code=True
        )
        print("Using the text-generation pipeline.")

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
    
            # Format the prompt to elicit an open-ended response
            prompt = f"""{instruction}

Question: "{question}"

Answer:"""
    
            try:
                response = self.pipeline(prompt, max_new_tokens=max_tokens, max_length=1024, num_return_sequences=1, temperature=0.2, top_p=0.9, do_sample=True)

                generated_text = response[0]["generated_text"].strip()

                # Ensure extraction only grabs the assistant's response
                if "Answer:" in generated_text:
                    final_answer = generated_text.split("Answer:")[-1].strip()
                else:
                    # If unexpected format, return as is
                    final_answer = generated_text
            
                return final_answer

            except Exception as e:
                print(f"Error occurred while processing the input: {input_text}")
                print(f"Error details: {str(e)}")
                return None

        else:  # Default to handling multiple-choice questions (for qa and fib_closed)
            options = re.findall(r"[أبجد]\.\s*[^\n]+", input_text)
            #options = re.findall(r"[ABCD]\.\s*[^\n]+", input_text)
    
            if not options:
                print("No options found in the input text.")
                return "Invalid input format."
    
            question = input_text.split(options[0])[0].strip()
    
            prompt = f"""{instruction}

    
    Question: {question}
    
    Options:
    {chr(10).join(options)}
    A: """
    
            try:
                # Clear CUDA cache before generation
                torch.cuda.empty_cache()
                
                # Generate with more conservative parameters
                response = self.pipeline(
                    prompt, 
                    max_new_tokens=min(max_tokens, 50),  # Limit to prevent memory issues
                    max_length=None,  # Don't set both max_length and max_new_tokens
                    num_return_sequences=1, 
                    temperature=0.2, 
                    top_p=0.9, 
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,  # Only return generated part
                    clean_up_tokenization_spaces=True
                )
    
                generated_text = response[0]["generated_text"].strip()
                
                
                # Extract only the letter after "A:" - handle newlines and spaces
                # Find the LAST occurrence of "A:" to avoid matching examples
                last_a_pos = generated_text.rfind("A:")
                if last_a_pos != -1:
                    # Look for pattern only after the last "A:"
                    after_last_a = generated_text[last_a_pos:]
                    pattern = r"A:\s*\n?\s*[\"']?([أبجد])"
                    match = re.search(pattern, after_last_a)
                    if match:
                        final_answer = match.group(1)
                        print(f"Pattern matched: {final_answer}")
                    else:
                        # Fallback: look for any Arabic letter at the very end
                        end_pattern = r"([أبجد])'?\s*$"
                        end_match = re.search(end_pattern, generated_text)
                        if end_match:
                            final_answer = end_match.group(1)
                            print(f"End pattern matched: {final_answer}")
                        else:
                            final_answer = generated_text
                            print("No pattern matched")
                else:
                    # Fallback if no "A:" found
                    end_pattern = r"([أبجد])'?\s*$"
                    end_match = re.search(end_pattern, generated_text)
                    if end_match:
                        final_answer = end_match.group(1)
                        print(f"End pattern matched: {final_answer}")
                    else:
                        final_answer = generated_text
                        print("No A: found")
    
                return final_answer
    
            except Exception as e:
                print(f"Error occurred while processing the input: {input_text}")
                print(f"Error details: {str(e)}")
                return None