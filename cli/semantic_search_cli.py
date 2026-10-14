import argparse
from semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_command, semantic_chunk_command, embed_chunks_command, search_chunked_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Search movies using keywords")
    # verify_parser.add_argument("query", type=str, help="Search query")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for a given term"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Text input to be convereted to embedding"
    )
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for a given term"
    )

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embeddings for a given term")
    embed_query_parser.add_argument("query", type=str, help="Query string for conversion")
    
    search_query_parser = subparsers.add_parser("search", help="Search and find similar documents")
    search_query_parser.add_argument("query", type=str, help="The query to be matched")
    search_query_parser.add_argument("--limit", type=int, help="Number of results",  default=5)

    chunk_parser = subparsers.add_parser("chunk", help="Generate document chunks")
    chunk_parser.add_argument("text", type=str, help="Input text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Option value for chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Option value for chunk size")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunking")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, help="Chunk overlap value", default=0)

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Created chunk embeddings")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search chunks")
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "":
            parser.print_help()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)            
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)


if __name__ == "__main__":
    main()
