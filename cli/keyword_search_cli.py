import argparse
import string
from lib.keyword_search import (
    has_matching_tokens,
    load_movies,
    load_stop_words,
    tokenize_text,
)

from lib.inverted_index import (
    index,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25tf_command,
    BM25_K1,
    BM25_B,
    bm25search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser(
        "build", help="Build inverted index and save to disk"
    )
    tf_parser = subparsers.add_parser(
        "tf", help="Print term frequency in document for a given ID"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Query token")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate TFIDF given document ID and query term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Query Term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Result limit"
    )
    args = parser.parse_args()

    match args.command:
        case "search":
            search_query = args.query
            query_tokens = tokenize_text(search_query)
            try:
                index.load()
            except FileNotFoundError as error:
                print(f"Failed to load index: {error}")
                return
            print(f"Searching for: {search_query}")
            movies = load_movies()
            for query in query_tokens:
                movie_ids = index.get_documents(query)
                for movie_id in movie_ids[:5]:
                    movie = index.docmap[movie_id]
                    print(f"{movie['id']} {movie['title']}")
            pass
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            bm25search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
