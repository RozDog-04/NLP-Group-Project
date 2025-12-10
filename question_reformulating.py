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
        Your job is to create lexical rewrites of the following question {n} different ways, keeping the meaning exactly
        the same. The rewrites will be used for lexical (BM25) search.

        Guidelines:
        - Keep all important names and nouns exactly as they appear.
        - Avoid pronouns like "he", "she", "they", "it".
        - Keep the sentence structure simple.
        - Keep the length under 20 words.
        - Output ONE rewrite per line with NO bullets, numbering, or extra text.

        Question:
        "{question}"
        """
        resp = self.llm.complete(prompt)
        rewrites = [ln.strip("- ").strip() for ln in resp.splitlines() if ln.strip()]
        return rewrites[:n]

    # 2) Semantic decomposition
    def semantic_decomposition(self, question: str, max_steps: int = 3) -> List[str]:
        prompt = f"""
        You are decomposing complex questions for multi-hop question answering over Wikipedia.

        Your task is to split the original question into up to {max_steps} atomic subquestions
        that correspond to the underlying reasoning steps.

        Definitions:
        - A subquestion is a smaller question whose answer is needed as part of a reasoning chain
          to answer the original question.
        - Each subquestion should be focused on ONE concrete fact (e.g. an entity's attribute,
          relation, or link to another entity).

        Guidelines:
        - Subquestions must be SELF-CONTAINED: repeat entity names explicitly, do not use pronouns.
        - Do NOT introduce new entity names that are not mentioned or clearly implied in the original question.
          When you need to refer to an unknown entity, describe it (e.g. "the author of Harry Potter'").
        - Each subquestion should be answerable from a short Wikipedia passage.
        - Each subquestion must be under 20 words.
        - Start from the first step in the reasoning chain and proceed in logical order.
        - Do NOT just paraphrase the original question.
        - Stop when you have enough subquestions to answer the original question, up to {max_steps}.
        - OUTPUT FORMAT: ONE subquestion per line, with NO bullets, numbers, or extra text.

        Example 1:
        Original question:
        "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who Matt Groening named after who?"

        Good subquestions:
        "Which 'The Simpsons' character did musician and satirist Allie Goertz write a song about?"
        "Who did Matt Groening name the character Milhouse after?"

        Example 2:
        Original question:
        "Were Scott Derrickson and Ed Wood of the same nationality?"

        Good subquestions:
        "What is the nationality of Scott Derrickson?"
        "What is the nationality of Ed Wood?"

        Now decompose the following question into subquestions:

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
        works, events, dates, objects etc.) mentioned in this question.

        For each entity you identify, write ONE query suitable for a Wikipedia search that focuses on
        the background or a key fact about that entity.

        Format:
        <entity>: <entity-focused query>

        Guidelines:
        - Include the entity name exactly.
        - Optionally mention a relevant relation from the question.
        - Keep each query under 20 words.
        - OUTPUT FORMAT: ONE subquestion per line, with NO bullets, numbers, or extra text.
        
        Example:
        Question:
        "Were Scott Derrickson and Ed Wood of the same nationality?"

        Good output:
        Scott Derrickson: Scott Derrickson nationality
        Ed Wood: Ed Wood nationality

        Now process this question:
        "{question}"
        """


        resp = self.llm.complete(prompt)
        queries: List[str] = [  ]
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
