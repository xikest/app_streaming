from openai import OpenAI

class GPTAssistant:
    def __init__(self, api_key, default_model="gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = default_model

    def generate_response(self, messages):
        full_response = ""
        for response in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.8
        ):
            full_response += (response.choices[0].delta.content or "")
        return full_response