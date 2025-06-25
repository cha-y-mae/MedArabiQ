import google.generativeai as genai
import os
import re

class GeminiHandler:
    def __init__(self, api_key=None, model="gemini-1.5-pro"):
        """
        Initialize the Gemini handler.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("API key for Gemini is missing. Set GEMINI_API_KEY in your environment.")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)

    def prompt(self, question, system_prompt, task=" "):
        """
        Generate a response from the Gemini model for a given question.
        """
        try:
            full_prompt = f"{system_prompt}\n\n{question}"

            response = self.client.generate_content(
                contents=[full_prompt], 
                generation_config={
                "temperature": 1.0,  # More deterministic output
                "max_output_tokens": 100  # optional: controls response length
    }
            )

            # Ensure raw_output is always assigned
            raw_output = response.text.strip() if hasattr(response, "text") and response.text else "No response generated."

            # Extract only the letter if the task is multiple choice
            if task in ["qa", "fib_closed"]:  
                match = re.search(r"\b([أبجد])\b", raw_output)
                if match:
                    return match.group(1)  # Extracted letter (أ, ب, ج, د)

                print(f"Unexpected output from Gemini (MCQ mode)")  # Debugging
                return "Invalid format"  # Fallback if no correct letter is found
            
            # For other tasks, return the full generated response**
            return raw_output


        except Exception as e:
            print(f"Error occurred while processing the question: {question}")
            print(f"Error details: {str(e)}")
            return None
