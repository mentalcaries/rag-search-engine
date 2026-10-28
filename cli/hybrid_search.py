import os
from dotenv import load_dotenv
from openai import OpenAI

from lib.inverted_index import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from lib.utils import load_movies
from lib.query_enhancement import enhance_query, rank_query, batch_rank_query
import json
from sentence_transformers import CrossEncoder


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
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        scores = {}
        for index, (doc_id, score) in enumerate(bm25_results, start=1):
            if doc_id not in scores:
                scores[doc_id] = {
                    "title": self.idx.docmap[doc_id]["title"],
                    "document": self.idx.docmap[doc_id],
                    "bm25_rank": index,
                    "semantic_rank": None,
                    "rrf_score": rrf_score(index),
                }
        for index, result in enumerate(semantic_search_results, start=1):
            if result["id"] in scores:
                scores[result["id"]]["semantic_rank"] = index
                scores[result["id"]]["rrf_score"] += rrf_score(index)
            else:
                scores[result["id"]] = {
                    "title": self.idx.docmap[doc_id]["title"],
                    "document": self.idx.docmap[doc_id],
                    "semantic_rank": index,
                    "rrf_score": rrf_score(index),
                    "bm25_rank": None,
                }

        sorted_scores = dict(
            sorted(
                scores.items(),
                key=lambda item: item[1]["rrf_score"],
                reverse=True,
            )[:limit]
        )
        return sorted_scores


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


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


def rrf_search_command(query, k=60, limit=25, enhance=None, rerank_method=None):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    query_limit = limit
    if rerank_method:
        query_limit = limit * 5

    results = hybrid_search.rrf_search(query, k, query_limit)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    if rerank_method == "individual":
        for result in results.values():
            rank = rank_query(query, result)
            result["rank"] = rank

        sorted_results = sorted(
            results.values(),
            key=lambda item: item["rank"],
            reverse=True,
        )
        results = sorted_results

    if rerank_method == "batch":
        llm_ranked_ids = batch_rank_query(query, json.dumps(results))
        llm_sorted_results = []
        for rank, doc_id in enumerate(llm_ranked_ids, start=1):
            result = results[doc_id]
            result["rank"] = rank
            llm_sorted_results.append(result)
        results = llm_sorted_results

    if rerank_method == "cross_encoder":
        pairs = []
        for doc in results.values():
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        scores = cross_encoder.predict(pairs)
        for doc, score in zip(results.values(), scores):
            doc["cross_encoder_score"] = score

        sorted_results = sorted(
            results.values(),
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )
        results = sorted_results

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "results": results[:limit],
    }
