from transformers import pipeline, AutoTokenizer

def generate_response(model_name, question, max_tokens=50):
    """
    Generate a response for a given question using a text-generation model.
    Args:
        model_name (str): The Hugging Face model name.
        question (str): The input question.
        max_tokens (int): Maximum tokens for the output.
    Returns:
        str: The predicted answer or raw output.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
    )

    # Set padding token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as a fallback

    # Load the text-generation pipeline
    qa_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
    )

    # Generate the response
    response = qa_pipeline(
        question,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    # Extract and return generated text
    return response[0]["generated_text"].strip()


# Example question
model_name = "inceptionai/jais-13b"  # Replace with your model name
question = "What is the capital of France?"
result = generate_response(model_name, question)
print("Answer:", result)

# Save the answer to a text file
with open('answer.txt', 'w', encoding='utf-8') as f:
    f.write(result)
