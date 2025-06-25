from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "inceptionai/jais-13b-chat"

try:
    # Load from cache only, no internet call
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        cache_dir="/scratch/ca2627/huggingface",  # your cache path
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        cache_dir="/scratch/ca2627/huggingface"
    )
    print("✅ Model and tokenizer loaded successfully from cache.")
except Exception as e:
    print("❌ Could not load from cache only.")
    print("Error:", e)
