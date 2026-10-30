import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .utils import load_dataset, load_movies


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "nvidia/nemotron-3.5-super-vl:free"


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    if k <= 0:
        return 0.0
    return sum(doc in relevant_docs for doc in retrieved_docs[:k]) / k


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    if not relevant_docs:
        return 0.0
    return sum(doc in relevant_docs for doc in retrieved_docs[:k]) / len(relevant_docs)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_command(limit: int = 5) -> dict[str, dict]:
    """Evaluate RRF search against the golden dataset."""
    from hybrid_search import HybridSearch

    movies = load_movies()
    test_cases = load_dataset()
    hybrid_search = HybridSearch(movies)
    results_by_query: dict[str, dict] = {}

    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = [result["title"] for result in search_results]

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score(precision, recall),
            "retrieved": retrieved_docs,
            "relevant": test_case["relevant_docs"],
        }

    return results_by_query


def llm_judge_results(query: str, results: list[dict]) -> list[int]:
    formatted_results = format_results(results)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    print("Submitting request to LLM...")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    scores = json.loads((response.choices[0].message.content or "").strip())

    if not isinstance(scores, list) or len(scores) != len(results):
        raise ValueError("LLM evaluation must return one score for every search result")
    return [int(score) for score in scores]


def format_results(results: list[dict]) -> list[str]:
    return [f"{index}. {result['title']}" for index, result in enumerate(results, 1)]
