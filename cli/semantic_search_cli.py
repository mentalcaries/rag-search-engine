import argparse
from semantic_search import verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Search movies using keywords")
    # verify_parser.add_argument("query", type=str, help="Search query")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embeddings for a given term")
    embed_text_parser.add_argument("text", type=str, help="Text input to be convereted to embedding")

    args = parser.parse_args()

    match args.command:
        case "":
            parser.print_help()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)


if __name__ == "__main__":
    main()