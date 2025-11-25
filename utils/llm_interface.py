import google.generativeai as genai
from openai import OpenAI
import json
import os

class LLMInterface:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        elif provider == "OpenRouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.model_name = "openai/gpt-4o-mini"

    def analyze_papers(self, papers_df, user_interests):
        papers_context = ""
        for idx, row in papers_df.iterrows():
            papers_context += f"ID: {idx}\nTitle: {row['title']}\nAbstract: {row['abstract']}\nAuthors: {row['authors']}\n\n"

        prompt = f"""
        You are an expert research assistant. The user is interested in: "{user_interests}".
        
        Here are the top {len(papers_df)} most relevant papers I found based on semantic search:
        
        {papers_context}
        
        Please select the top 3-5 papers that are MOST relevant to the user's specific interest.
        
        For each paper, return a JSON object with the following fields:
        - "id": The ID of the paper provided in the context.
        - "title": The exact title of the paper.
        - "authors": A list of objects, where each object has:
            - "name": Author name.
        - "keywords": A list of 3-5 relevant keywords/tags for this paper.
        - "relevance": A 1-2 sentence explanation of why this paper is relevant to the user.
        - "icebreaker": A specific question the user could ask one of the authors.
        
        Return ONLY a valid JSON array of these objects. Do not include markdown formatting like ```json ... ```.
        """
        
        return self._generate(prompt)

    def _generate(self, prompt):
        try:
            if self.provider == "Google Gemini":
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    if "404" in str(e) or "not found" in str(e).lower():
                        try:
                            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            return f"Error: The selected model was not found. Your API key has access to: {', '.join(models)}. \n\nOriginal Error: {e}"
                        except Exception as list_e:
                            return f"Error generating response: {e}. (Also failed to list available models: {list_e})"
                    raise e
            elif self.provider == "OpenRouter":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "API key not valid" in error_str:
                return "Error: API key not valid. Please check your Google Gemini API key."
            return f"Error generating response: {error_str}"
