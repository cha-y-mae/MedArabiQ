from openai import OpenAI
import re

def extract_letter_from_text(text):
    """
    Extract a single Arabic MCQ letter from model-generated output.
    Handles formats like:
    - "الإجابة الصحيحة هي: ب"
    - "Correct letter: ب"
    - "ب"
    """
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"(الإجابة\s*(الصحيحة)?\s*(هي)?\s*[:：]?)", "", text)
    text = re.sub(r"(الجواب\s*(الصحيح)?\s*(هو)?\s*[:：]?)", "", text)
    text = text.strip()

    match = re.search(r"([أبجدهـه])", text)
    if match:
        letter = match.group(1)
        return 'هـ' if letter in {'ه', 'هـ'} else letter

    return None


class OpenAIHandler:
    def __init__(self, api_key, model="gpt-4-0613"):
        """
        Initialize the OpenAI handler.
        """
        self.client = OpenAI(api_key=api_key)  # Use the OpenAI client
        self.model = model

    def prompt(self, question, instruction, task_type=" "):
        """
        Generate a response from the OpenAI model for a given question.
        """
        system_prompt = instruction
        try:
            # Create a chat completion using the OpenAI client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=1.0,  # Configure response randomness
                max_tokens=1000,
                top_p=1.0,        # Nucleus sampling
                frequency_penalty=0.0,  # Penalize repetition
                presence_penalty=0.0    # Encourage new topics
            )

            # Extract the message content from the response
            raw_output = response.choices[0].message.content.strip()
            return raw_output
            #return extract_letter_from_text(raw_output)

        except Exception as e:
            # Handle exceptions and log errors
            print(f"Error occurred while processing the question: {question}")
            print(f"Error details: {str(e)}")
            return None
