import os
import json
from pathlib import Path
from typing import List, Optional, Any

from mistralai import Mistral


def _load_mistral_api_key() -> Optional[str]:
    """
    Fetch MISTRAL_API_KEY from env or a local .env file.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return api_key

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            if key.strip() == "MISTRAL_API_KEY":
                api_key = val.strip().strip('"').strip("'")
                if api_key:
                    os.environ["MISTRAL_API_KEY"] = api_key
                    return api_key
    return None


class AnswerGenerator:
    def __init__(self, model: str = "mistral-small-latest", system_prompt: Optional[str] = None):
        api_key = _load_mistral_api_key()
        if not api_key:
            raise RuntimeError("Please set MISTRAL_API_KEY in your environment.")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a QA model for the HotpotQA dataset.\n"
            "- Always answer with ONLY the final answer text.\n"
            "- For yes/no questions, answer exactly 'yes' or 'no' (lowercase).\n"
            "- For other questions, answer with a short noun phrase only "
            "(e.g. 'Animorphs', 'Chief of Protocol of the United States').\n"
            "- Do NOT write full sentences or explanations.\n"
            "- Use only the provided context; if the answer is not supported, reply exactly:\n"
            "  I cannot answer from the given context."
        )

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = "\n\n---\n\n".join(contexts)
        return (
            "You are given a HotpotQA question and some context passages.\n"
            "Return ONLY the final answer.\n"
            "If yes/no, answer exactly 'yes' or 'no'. Otherwise, return a short phrase.\n\n"
            f"Question:\n{question}\n\n"
            f"Context passages:\n{context_block}\n\n"
            "Answer:"
        )

    @staticmethod
    def _normalize_yes_no(answer: str) -> str:
        ans = answer.strip().lower()
        if ans.startswith("yes"):
            return "yes"
        if ans.startswith("no"):
            return "no"
        return answer.strip()

    def generate_answer(self, question: str, contexts: List[str], return_prompt: bool = False):
        prompt = self._build_prompt(question, contexts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        res = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        answer = ""
        try:
            choice = res.choices[0]
            message = getattr(choice, "message", choice.get("message"))
            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    else:
                        text_val = getattr(block, "text", None)
                        parts.append(text_val if text_val is not None else str(block))
                content = "".join(parts)
            if isinstance(content, str):
                answer = content.strip()
            if not answer and content is not None:
                answer = str(content).strip()
        except Exception:
            try:
                answer = str(res.choices[0].message.content).strip()
            except Exception:
                answer = ""

        answer = self._normalize_yes_no(answer)

        if return_prompt:
            return answer, prompt
        return answer


    def score_candidate_answers(
        self,
        question: str,
        contexts: list[str],
        answers: list[str],
    ) -> list[float]:
        """
        Given a question, a list of context passages, and candidate answers,
        return confidence scores per answer in [0, 1], aligned with `answers`.
        """
        if not answers:
            return []

        context_block = "\n\n---\n\n".join(contexts)
        answers_block = "\n".join([f"[{i}] {ans}" for i, ans in enumerate(answers)])

        system_prompt = (
            "You are evaluating candidate answers for a HotpotQA question.\n"
            "Given:\n"
            "- the question\n"
            "- the supporting context passages\n"
            "- a list of candidate answers\n\n"
            "Rate how well each candidate answer is supported by the CONTEXT ONLY\n"
            "Use a score between 0.0 and 1.0:\n"
            "- 1.0 = clearly and directly supported by the context.\n"
            "- 0.8 = strongly supported, small ambiguity.\n"
            "- 0.5 = partially supported or plausible but not clearly stated.\n"
            "- 0.2 = weakly supported, only vague hints.\n"
            "- 0.0 = contradicted by the context or not supported at all.\n"
            "Return ONLY a JSON array of objects of the form:\n"
            '[{\"index\": 0, \"confidence\": 0.73}, {\"index\": 1, \"confidence\": 0.35}, ...]\n'
            "where `index` is the answer index and `confidence` is a float in [0, 1]."
        )

        user_content = (
            f"Question:\n{question}\n\n"
            f"Context passages:\n{context_block}\n\n"
            f"Candidate answers:\n{answers_block}\n\n"
            "JSON:"
        )

        res = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )

        content = ""
        try:
            choice = res.choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    else:
                        text_val = getattr(block, "text", None)
                        parts.append(text_val if text_val is not None else str(block))
                content = "".join(parts)
        except Exception:
            content = ""

        if not isinstance(content, str):
            content = str(content)

        scores = [0.0] * len(answers)

        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(content[start : end + 1])
                except Exception:
                    parsed = None

        if not parsed:
            return scores

        for entry in parsed:
            try:
                idx = int(entry.get("index"))
                conf = float(entry.get("confidence", 0))
                if 0 <= idx < len(answers):
                    scores[idx] = max(0.0, min(1.0, conf))
            except Exception:
                continue

        return scores


class QueryRewriter:
    """
    Optional helper to rewrite a question into retrieval-friendly queries.
    """

    def __init__(self, model: str = "mistral-small-latest"):
        api_key = _load_mistral_api_key()
        if not api_key:
            raise RuntimeError("Please set MISTRAL_API_KEY in your environment.")
        self.client = Mistral(api_key=api_key)
        self.model = model

    def _extract_content(self, res: Any) -> str:
        content = ""
        try:
            choice = res.choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    else:
                        text_val = getattr(block, "text", None)
                        parts.append(text_val if text_val is not None else str(block))
                content = "".join(parts)
        except Exception:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        return content.strip()

    def rewrite(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the question into a concise keyword search query. "
                    "Remove filler words; keep names, entities, and key nouns/verbs. "
                    "Return only the query string."
                ),
            },
            {"role": "user", "content": question},
        ]

        res = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        return self._extract_content(res) or question

    def rewrite_entity_focused(self, question: str) -> str:
        """
        Generate an entity-focused query variant from the original question.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the question into an entity-focused search query. "
                    "Extract key entities (people, places, organizations, works) and "
                    "include minimal relational terms if needed. "
                    "Return only the compact entity-focused query."
                ),
            },
            {"role": "user", "content": question},
        ]

        res = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        return self._extract_content(res) or question


class ContextReranker:
    def __init__(self, model: str = "mistral-small-latest"):
        api_key = _load_mistral_api_key()
        if not api_key:
            raise RuntimeError("Please set MISTRAL_API_KEY in your environment.")
        self.client = Mistral(api_key=api_key)
        self.model = model

    @staticmethod
    def _extract_json_block(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    def score(self, question: str, contexts: List[str]) -> List[float]:
        """
        Use an LLM to assign a confidence score to each context.
        Returns a list of floats in [0, 1], one per context.
        """
        if not contexts:
            return []

        numbered_contexts = "\n".join(
            [f"{idx}: {ctx}" for idx, ctx in enumerate(contexts)]
        )

        prompt = (
            "You are ranking passages for question answering.\n"
            "Given the question and the numbered contexts below, assign each context a "
            "confidence score between 0 and 1 for how useful it is to answer the question.\n"
            "Return ONLY a JSON array like:\n"
            '[{"index": 0, "confidence": 0.9}, {"index": 1, "confidence": 0.1}, ...]\n\n'
            f"Question:\n{question}\n\n"
            f"Contexts:\n{numbered_contexts}\n\n"
            "JSON:"
        )

        res = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": "Select contexts for QA"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
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

        parsed = self._extract_json_block(content) or []

        scores = [0.0] * len(contexts)
        for entry in parsed:
            try:
                idx = int(entry.get("index"))
                conf = float(entry.get("confidence", 0))
                if 0 <= idx < len(contexts):
                    scores[idx] = max(0.0, min(1.0, conf))
            except Exception:
                continue

        return scores

    def rerank(self, question: str, contexts: List[str]) -> List[int]:
        scores = self.score(question, contexts)
        return sorted(range(len(contexts)), key=lambda i: scores[i], reverse=True)
