import os

from lib.inverted_index import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from lib.utils import load_movies


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = []
        semantic_scores = []
        for doc_id, score in bm25_results:
            bm25_scores.append(score)
        normalized_bm25_scores = normalize_scores(bm25_scores)

        for result in semantic_search_results:
            semantic_scores.append(result["score"])
        normalized_semantic_scores = normalize_scores(semantic_scores)

        scores = {}

        for result, score in zip(bm25_results, normalized_bm25_scores):
            doc_id, _ = result
            scores[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_score": score,
                "semantic_score": 0.0,
                "hybrid_score": hybrid_score(score, 0.0, alpha),
            }

        for result, score in zip(semantic_search_results, normalized_semantic_scores):
            if result["id"] in scores:
                scores[result["id"]]["semantic_score"] = score
                scores[result["id"]]["hybrid_score"] = hybrid_score(
                    scores[result["id"]]["bm25_score"], score, alpha
                )
            else:
                scores[result["id"]] = {
                    "document": self.idx.docmap[result["id"]],
                    "bm25_score": 0.0,
                    "semantic_score": score,
                    "hybrid_score": hybrid_score(0.0, score, alpha),
                }
        sorted_scores = dict(
            sorted(
                scores.items(),
                key=lambda item: item[1]["hybrid_score"],
                reverse=True,
            )[:limit]
        )
        return sorted_scores

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores):
    if not scores:
        return
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)
    normalized = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized.append(normalized_score)
    return normalized


def weighted_search_command(query, alpha=0.5, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)
