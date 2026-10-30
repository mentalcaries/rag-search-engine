import json
import os

from sentence_transformers import CrossEncoder

from lib.evaluation import llm_judge_results
from lib.inverted_index import InvertedIndex
from lib.query_enhancement import (
    batch_rank_query,
    enhance_query,
    rank_query,
)
from lib.utils import load_movies
from semantic_search import ChunkedSemanticSearch


SEARCH_CANDIDATE_MULTIPLIER = 500
RERANK_CANDIDATE_MULTIPLIER = 5


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * SEARCH_CANDIDATE_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_CANDIDATE_MULTIPLIER
        )
        return combine_search_results(
            bm25_results, semantic_results, self.idx.docmap, alpha
        )[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * SEARCH_CANDIDATE_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_CANDIDATE_MULTIPLIER
        )
        return reciprocal_rank_fusion(bm25_results, semantic_results, self.idx.docmap, k)[
            :limit
        ]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[tuple[int, float]],
    semantic_results: list[dict],
    documents: dict[int, dict],
    alpha: float,
) -> list[dict]:
    """Combine normalized BM25 and semantic scores by document ID."""
    bm25_scores = normalize_scores([score for _, score in bm25_results])
    semantic_scores = normalize_scores([result["score"] for result in semantic_results])
    combined: dict[int, dict] = {}

    for (doc_id, _), score in zip(bm25_results, bm25_scores):
        combined[doc_id] = {
            "title": documents[doc_id]["title"],
            "document": documents[doc_id],
            "bm25_score": score,
            "semantic_score": 0.0,
            "hybrid_score": hybrid_score(score, 0.0, alpha),
        }

    for result, score in zip(semantic_results, semantic_scores):
        doc_id = result["id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "title": documents[doc_id]["title"],
                "document": documents[doc_id],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
                "hybrid_score": hybrid_score(0.0, 0.0, alpha),
            }

        combined[doc_id]["semantic_score"] = score
        combined[doc_id]["hybrid_score"] = hybrid_score(
            combined[doc_id]["bm25_score"], score, alpha
        )

    return sorted(combined.values(), key=lambda result: result["hybrid_score"], reverse=True)


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    semantic_results: list[dict],
    documents: dict[int, dict],
    k: int,
) -> list[dict]:
    """Fuse BM25 and semantic rankings using reciprocal rank fusion."""
    fused: dict[int, dict] = {}

    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        fused[doc_id] = {
            "title": documents[doc_id]["title"],
            "document": documents[doc_id],
            "bm25_rank": rank,
            "semantic_rank": None,
            "rrf_score": rrf_score(rank, k),
        }

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in fused:
            fused[doc_id] = {
                "title": documents[doc_id]["title"],
                "document": documents[doc_id],
                "bm25_rank": None,
                "semantic_rank": rank,
                "rrf_score": rrf_score(rank, k),
            }
        else:
            fused[doc_id]["semantic_rank"] = rank
            fused[doc_id]["rrf_score"] += rrf_score(rank, k)

    return sorted(fused.values(), key=lambda result: result["rrf_score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = 0.5, limit: int = 5
) -> list[dict]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str,
    k: int = 60,
    limit: int = 25,
    enhance: str | None = None,
    rerank_method: str | None = None,
    evaluate: bool = False,
) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    original_query = query
    enhanced_query = None

    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    candidate_limit = limit * RERANK_CANDIDATE_MULTIPLIER if rerank_method else limit
    results = hybrid_search.rrf_search(query, k, candidate_limit)

    if rerank_method == "individual":
        for result in results:
            result["rank"] = rank_query(query, result)
        results = sorted(results, key=lambda item: item["rank"], reverse=True)

    elif rerank_method == "batch":
        results_by_id = {result["document"]["id"]: result for result in results}
        ranked_ids = batch_rank_query(query, json.dumps(results))
        results = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in results_by_id:
                result = results_by_id[doc_id]
                result["rank"] = rank
                results.append(result)

    elif rerank_method == "cross_encoder":
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        pairs = [
            [
                query,
                f"{result['document']['title']} - {result['document']['description']}",
            ]
            for result in results
        ]
        scores = cross_encoder.predict(pairs)
        for result, score in zip(results, scores):
            result["cross_encoder_score"] = score
        results = sorted(
            results, key=lambda item: item["cross_encoder_score"], reverse=True
        )

    results = results[:limit]
    evaluation_scores = llm_judge_results(query, results) if evaluate else None

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "evaluation_scores": evaluation_scores,
        "results": results,
    }
