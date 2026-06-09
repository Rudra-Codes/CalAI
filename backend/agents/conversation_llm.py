class ConversationalAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.response_prompt = """You are a nutrition analysis expert.
Analyze foods, meals, ingredients, calories, macros, and nutrition facts accurately.
State assumptions when estimates are uncertain.
Answer all other user questions helpfully, clearly, and concisely. Prioritize accuracy, safety, and practical guidance.
"""
    def __call__(self, query=None):
        try:
            response = self.llm_client.chat.completions.create(
                # model= "meta-llama/llama-4-scout-17b-16e-instruct",
                model= "llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.response_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling LLM: {e}")
            return None