from transformers import pipeline, AutoTokenizer
import re

def generate_response(model_name, prompt, max_tokens=20):
    
    # Load the tokenizer manually
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map="auto"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as a fallback

    qa_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
    )

    
    
    response = qa_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    
    # Extract generated text
    generated_text = response[0]["generated_text"]
    
    # Parse Arabic options
    options = ["أ. بلوتو", "ب. زحل", "ج. جهنم", "د. الأرض"]
    
    for option in options:
        if option in generated_text:
            return option
    
    return generated_text

# Arabic prompt example
model_name = "inceptionai/jais-13b"
prompt = "س: أي من هذه ليس كوكباً؟ أ. بلوتو ب. زحل ج. جهنم د. الأرض\n\nالإجابة الدقيقة (الحرف والنص):"
result = generate_response(model_name, prompt)
print("الإجابة:", result)

with open('answerr.txt', 'w', encoding='utf-8') as f:
    f.write(result)