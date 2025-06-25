from openai import OpenAI

class DeepSeekHandler:
    def __init__(self, api_key, model="deepseek-chat"):
        """
        Initialize the DeepSeek handler using OpenAI-compatible API.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"  # DeepSeek's API endpoint
        )
        self.model = model  # "deepseek-chat" invokes DeepSeek-V3

    def prompt(self, question, instruction, task_type=" "):
        """
        Generate a response from the DeepSeek model for a given question.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question}
                ],
                temperature=0.4,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error occurred while processing the question: {question}")
            print(f"Error details: {str(e)}")
            return None
