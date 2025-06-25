import anthropic
from models.model_handler_base import ModelHandlerBase
import re

def extract_letter_from_text(text):
        """
        Clean and extract a single Arabic MCQ letter from model-generated output.
        Handles noisy output like:
        - "الإجابة الصحيحة هي: ب. ..."
        - "الإجابة: ج"
        - "ج. يسمى هذا ..."
        """
        if not text:
            return None
    
        # Normalize and remove leading Arabic answer patterns
        text = text.strip()
        text = re.sub(r"(الإجابة\s*(الصحيحة)?\s*(هي)?\s*[:：]?)", "", text)
        text = re.sub(r"(الجواب\s*(الصحيح)?\s*(هو)?\s*[:：]?)", "", text)
        text = text.strip()
    
        # Remove all Arabic text before first letter (if still present)
        match = re.search(r"([أبجدهـه])", text)
        if match:
            letter = match.group(1)
            return 'هـ' if letter in {'ه', 'هـ'} else letter
    
        return None


class Claude35SonnetHandler(ModelHandlerBase):
    """
    Handler for Anthropic's Claude 3.5 Sonnet model (post-March 2024 API).
    """
    def __init__(self, api_key, model_name="claude-3-5-sonnet-20240620"):
        self.client = anthropic.Anthropic(api_key="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")
        self.model_name = model_name

    def prompt(self, input_text, task=" ", system_prompt="This is a multiple-choice question, choose the correct option. The output should consist only of the single letter of the correct answer with no explanation", **kwargs):
        max_tokens = 5 if task == "fib_open" else kwargs.get("max_tokens", 100)
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.4),
                system=system_prompt,
                messages=[{"role": "user", "content": input_text}]
            )
    
            raw_output = response.content[0].text.strip() if response.content else "No response generated."
            return raw_output
            #extract_letter_from_text(raw_output)  # <- crucial
    
        except Exception as e:
            print(f"Error during Claude prompt: {e}")
            return None

    

