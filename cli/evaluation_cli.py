import argparse
from lib.utils import load_dataset, load_movies
from hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_cases = load_dataset()
    movies = load_movies()
    search = HybridSearch(movies)

    print("k=\n", limit)
    for test_case in test_cases:
        results = search.rrf_search(test_case["query"], 60, limit)
        retrieved_titles = [r["title"] for r in results.values()]

        overlap = list(set(retrieved_titles) & set(test_case["relevant_docs"]))
        precision = len(overlap) / len(retrieved_titles)

        print(f"- Query: {test_case['query']}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Retrieved: {retrieved_titles}")
        print(f"    - Relevant: {test_case['relevant_docs']}")


if __name__ == "__main__":
    main()
