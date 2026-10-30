import argparse

from hybrid_search import normalize_scores, weighted_search_command, rrf_search_command


def print_weighted_results(results: list[dict]) -> None:
    for index, result in enumerate(results, start=1):
        document = result["document"]
        print(f"{index}. {document['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(
            f"  BM25: {result['bm25_score']:.3f}, "
            f"Semantic: {result['semantic_score']:.3f}"
        )
        print(f"  {document['description'][:100]}...")


def print_rrf_results(result: dict) -> None:
    if result["enhanced_query"]:
        print(
            f"Enhanced query ({result['enhance_method']}): "
            f"'{result['original_query']}' -> '{result['enhanced_query']}'\n"
        )

    if result["evaluation_scores"] is not None:
        for index, (score, search_result) in enumerate(
            zip(result["evaluation_scores"], result["results"]), start=1
        ):
            print(f"{index}. {search_result['document']['title']}: {score}/3")
        return

    if result["rerank_method"] in {"individual", "batch"}:
        print(
            f"Re-ranking top {len(result['results'])} results using "
            f"{result['rerank_method']} method..."
        )

    print(f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):")
    for index, search_result in enumerate(result["results"], start=1):
        document = search_result["document"]
        print(f"{index}. {document['title']}")

        if result["rerank_method"] == "individual" and search_result.get("rank"):
            print(f"    Re-rank Score: {search_result['rank']}")
        elif result["rerank_method"] == "batch" and search_result.get("rank"):
            print(f"    Re-rank Rank: {search_result['rank']}")
        elif result["rerank_method"] == "cross_encoder":
            print(
                f"    Cross Encoder Score: "
                f"{search_result['cross_encoder_score']:.3f}"
            )

        print(f"   RRF Score: {search_result['rrf_score']:.3f}")
        ranks = []
        if search_result.get("bm25_rank"):
            ranks.append(f"BM25 Rank: {search_result['bm25_rank']}")
        if search_result.get("semantic_rank"):
            ranks.append(f"Semantic Rank: {search_result['semantic_rank']}")
        if ranks:
            print(f"   {', '.join(ranks)}")
        print(f"   {document['description'][:100]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("scores", nargs="*", type=float)

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument("--evaluate", action="store_true")
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"]
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            print_weighted_results(results)

        case "rrf-search":
            result = rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )

            print_rrf_results(result)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
