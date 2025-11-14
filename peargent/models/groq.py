#peargent/models/groq.py

import os
import requests
import json
from typing import Optional, Dict, Iterator
from .base import BaseModel

class GroqModel(BaseModel):
    ENDPOINT_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        parameters: Optional[Dict] = None,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        super().__init__(model_name, parameters)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Groq API key not found. Set GROQ_API_KEY in environment or pass `api_key=`.")
        
        self.endpoint_url = endpoint_url or self.ENDPOINT_URL

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.parameters.get("system_prompt", "")},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.parameters.get("temperature", 0.7),
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "tool_choice": "none"
        }
        
        response = requests.post(self.ENDPOINT_URL, headers=headers, json=body)
        
        if response.status_code != 200:
            raise RuntimeError(f"Groq API error: {response.status_code}, {response.text}")
            # return f"Groq API error: {response.status_code}, {response.text}"

        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream text completion from Groq API.

        Args:
            prompt: The input prompt

        Yields:
            String chunks of the generated text as they arrive
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.parameters.get("system_prompt", "")},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.parameters.get("temperature", 0.7),
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "tool_choice": "none",
            "stream": True  # Enable streaming
        }

        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=body,
            stream=True  # Enable streaming in requests
        )

        if response.status_code != 200:
            raise RuntimeError(f"Groq API error: {response.status_code}, {response.text}")

        # Process the streaming response
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')

                # Skip empty lines and comments
                if not line_str.strip() or line_str.startswith(':'):
                    continue

                # Remove "data: " prefix
                if line_str.startswith('data: '):
                    line_str = line_str[6:]

                # Check for end of stream
                if line_str.strip() == '[DONE]':
                    break

                try:
                    chunk_data = json.loads(line_str)

                    # Extract content from the chunk
                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                        delta = chunk_data['choices'][0].get('delta', {})
                        content = delta.get('content', '')

                        if content:
                            yield content

                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue