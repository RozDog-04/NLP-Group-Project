from typing import Optional
from mistralai import Mistral
from llm_pipeline import _load_mistral_api_key


class MistralCompleter:
    """
    Small helper for generic prompts (question rewrites, decompositions, entity-focused queries).
    """

    def __init__(self, model: str = "mistral-small-latest", system_prompt: Optional[str] = None):
        api_key = _load_mistral_api_key()
        if not api_key:
            raise RuntimeError("No API key found")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant for query reformulation for a RAG system using BM25S retriever."

    def complete(self, user_prompt: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        res = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        choice = res.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                else:
                    text_val = getattr(block, "text", None)
                    parts.append(text_val if text_val is not None else str(block))
            content = "".join(parts)

        if not isinstance(content, str):
            content = str(content)

        return content.strip()
