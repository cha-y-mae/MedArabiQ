from transformers import pipeline, AutoTokenizer

def generate_response(model_name, question, max_tokens=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qa_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
    )

    model = qa_pipeline.model
    print("Model configuration:")
    print(model.config)

    # Tokenize the input
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    print("Tokenized input:", inputs)
    print("Decoded tokens:", tokenizer.decode(inputs["input_ids"][0]))

    # Generate the response
    response = qa_pipeline(
        question,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    print("Pipeline output:", response)
    return response[0]["generated_text"].strip()

model_name = "inceptionai/jais-13b"
question = "What is the capital of France?"
result = generate_response(model_name, question)
print("Answer:", result)
