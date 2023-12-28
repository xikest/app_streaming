from openai import OpenAI

class GPTAssistant:
    def __init__(self, api_key, role:str=None, temp=0.8, default_model="gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = default_model
        self.role = role
        self.temp=temp

    def generate_response(self, messages):

        if self.role is not None:
            initial_role = {"role": "system", "content": self.role}
            messages = [initial_role] + messages
            self.role = None

        full_response = ""
        for response in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temp
        ):
            full_response += (response.choices[0].delta.content or "")
        return full_response
