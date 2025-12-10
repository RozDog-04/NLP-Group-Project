from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from BM25S_retrieval import BM25Retriever
from question_reformulating import QuestionRewriter


class MultiTrajectoryBM25Retriever:
    """
    Runs BM25S retrieval for multiple trajectories:
      - 'original': the original question
      - 'rewrite' : simple rewrites
      - 'decomp'  : semantic decompositions
      - 'entity'  : entity-focused queries
    """

    def __init__(
        self,
        index_path: str = "data/index/bm25s_index",
        store_path: str = "data/index/bm25_store.pkl",
        max_workers: int = 4,
    ):
        self.bm25 = BM25Retriever(index_path=index_path, store_path=store_path)
        self.max_workers = max_workers

    def _retrieve_for_queries(
        self, queries: List[str], top_k_per_query: int
    ) -> List[Dict[str, Any]]:
        """
        Runs BM25 for each query in `queries` and merges the results
        If a document appears for multiple queries it will keep the highest score
        """
        doc_best: Dict[int, Dict[str, Any]] = {}

        for q in queries:
            q = q.strip()
            if not q:
                continue
            results, _ = self.bm25.retrieve(q, top_k=top_k_per_query)
            for r in results:
                doc_id = r["doc_id"]
                score = r["score"]
                if doc_id not in doc_best or score > doc_best[doc_id]["score"]:
                    doc_best[doc_id] = r

        return sorted(doc_best.values(), key=lambda d: d["score"], reverse=True)

    def multi_trajectory_retrieve(
        self,
        question: str,
        rewriter: QuestionRewriter,
        top_k_per_query: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
        {
          "original": {"queries": [question],         "docs": [...]},
          "rewrite":  {"queries": [q1, q2, ...],      "docs": [...]},
          "decomp":   {"queries": [subq1, ...],       "docs": [...]},
          "entity":   {"queries": [entity_q1, ...],   "docs": [...]},
        }
        """
        # Builds query sets
        original_queries = [question]
        rewrite_queries  = rewriter.simple_rewrites(question, n=3)
        decomp_queries   = rewriter.semantic_decomposition(question, max_steps=3)
        entity_queries   = rewriter.entity_focused(question, max_entities=3)

        trajectories: Dict[str, List[str]] = {
            "original": original_queries,
            "rewrite":  rewrite_queries,
            "decomp":   decomp_queries,
            "entity":   entity_queries,
        }

        results: Dict[str, Dict[str, Any]] = {}

        # Dispatches retrieval in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_name = {
                pool.submit(self._retrieve_for_queries, qlist, top_k_per_query): name
                for name, qlist in trajectories.items()
                if qlist
            }

            for fut, name in [(f, n) for f, n in future_to_name.items()]:
                docs = fut.result()
                results[name] = {"queries": trajectories[name], "docs": docs}

        return results
