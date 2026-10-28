import argparse

from hybrid_search import normalize_scores, weighted_search_command, rrf_search_command


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
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"]
    )

    args = parser.parse_args()

    match args.command:
        case "":
            parser.print_help()
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for index, (_, result) in enumerate(results.items(), start=1):
                document = result["document"]

                print(f"{index}. {document['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {document['description'][:100]}...")
        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): "
                    f"'{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )
            if args.rerank_method == "individual":
                print(
                    f"Re-ranking top {len(result['results'])} results using individual method..."
                )

            if args.rerank_method == "batch":
                print(
                    f"Re-ranking top {len(result['results'])} results using batch method..."
                )

            print(
                f"Reciprocal Rank Fusion Results for "
                f"'{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['document']['title']}")

                if args.rerank_method == "individual" and res.get("rank"):
                    print(f"    Re-rank Score: {res['rank']}/{len(result['results'])}")

                if args.rerank_method == "batch" and res.get("rank"):
                    print(f"    Re-rank Rank: {res['rank']}")
                if args.rerank_method == "cross_encoder" and res.get(
                    "cross_encoder_score"
                ):
                    print(
                        f"    Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )

                print(f"   RRF Score: {res['rrf_score']:.3f}")

                ranks = []

                if res.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {res['bm25_rank']}")

                if res.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {res['semantic_rank']}")

                if ranks:
                    print(f"   {', '.join(ranks)}")

                print(f"   {res['document']['description'][:100]}...")
                print()


if __name__ == "__main__":
    main()
