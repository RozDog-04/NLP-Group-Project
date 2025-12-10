from typing import List
from llm_query_utils import MistralCompleter


class QuestionRewriter:
    """
    Generates multiple query variations of a question:
    - original form
    - simple lexical rewrites
    - semantic decompositions
    - entity-focused queries

    All prompts are tuned for the BM25 retrieval.
    """

    def __init__(self, completer: MistralCompleter | None = None):
        self.llm = completer or MistralCompleter()

    # 1) Simple lexical rewrites
    def simple_rewrites(self, question: str, n: int = 2) -> List[str]:
        prompt = f"""
        Rewrite the following question {n} different ways, keeping the meaning exactly
        the same. The rewrites will be used for lexical (BM25) search.

        Guidelines:
        - Keep all important names and nouns exactly as they appear.
        - Avoid pronouns like "he", "she", "they", "it".
        - Keep the sentence structure simple.
        - Keep the length under 20 words.
        - Output ONE rewrite per line with NO bullets or numbering.

        Question:
        "{question}"
        """
        resp = self.llm.complete(prompt)
        rewrites = [ln.strip("- ").strip() for ln in resp.splitlines() if ln.strip()]
        return rewrites[:n]

    # 2) Semantic decomposition
    def semantic_decomposition(self, question: str, max_steps: int = 3) -> List[str]:
        prompt = f"""
        The goal is multi-hop question answering over Wikipedia.

        Decompose the question into up to {max_steps} subquestions that, if answered in
        order, would allow you to answer the original question.

        Guidelines:
        - Each subquestion must be self-contained (repeat entity names explicitly).
        - Keep each subquestion under 20 words.
        - Output ONE subquestion per line with NO bullets or numbering.

        Original question:
        "{question}"
        """
        resp = self.llm.complete(prompt)
        subs = [ln.strip("- ").strip() for ln in resp.splitlines() if ln.strip()]
        return subs[:max_steps]

    # 3) Entity-focused queries
    def entity_focused(self, question: str, max_entities: int = 3) -> List[str]:
        
        prompt = f"""
        Identify up to {max_entities} important entities (people, places, organizations,
        works, events, objects etc.) mentioned in this question.

        For each entity you identify, write ONE query suitable for a Wikipedia search that focuses on
        the background or key fact about that entity.

        Format:
        <entity>: <entity-focused query>

        Guidelines:
        - Include the entity name exactly.
        - Optionally mention a relevant relation from the question.
        - Keep each query under 20 words.
        - Return ONE entity/query per line.

        Question:
        "{question}"
        """

        resp = self.llm.complete(prompt)
        queries: List[str] = []
        for ln in resp.splitlines():
            ln = ln.strip()
            if ":" not in ln:
                continue
            _entity, q = ln.split(":", 1)
            queries.append(q.strip())
        return queries[:max_entities]


# Simple CLI so I can manually test the question reformulating
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tests question reformulation for MT-RAG-EV.")
    parser.add_argument("--question", type=str, required=True, help="Inputs question to reformulate")
    parser.add_argument("--n-rewrites", type=int, default=3)
    parser.add_argument("--n-decomp", type=int, default=3)
    parser.add_argument("--n-entities", type=int, default=3)

    args = parser.parse_args()

    rewriter = QuestionRewriter()

    print("=" * 80)
    print("Original question:")
    print(args.question)
    print()

    rewrites = rewriter.simple_rewrites(args.question, n=args.n_rewrites)
    print("=" * 80)
    print("Simple rewrites:")
    for i, q in enumerate(rewrites, 1):
        print(f"{i}. {q}")
    print()

    decomps = rewriter.semantic_decomposition(args.question, max_steps=args.n_decomp)
    print("=" * 80)
    print("Semantic decompositions:")
    for i, q in enumerate(decomps, 1):
        print(f"{i}. {q}")
    print()

    entities = rewriter.entity_focused(args.question, max_entities=args.n_entities)
    print("=" * 80)
    print("Entity-focused queries:")
    for i, q in enumerate(entities, 1):
        print(f"{i}. {q}")
    print()
