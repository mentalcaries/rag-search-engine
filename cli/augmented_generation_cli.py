import argparse
from augmented_generation import rag_command, summarize_command, citations_command, question_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser = subparsers.add_parser("summarize", help="Summarize query results")
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("--limit", type=int, default=5)

    citations_parser = subparsers.add_parser("citations", help="Provide citations for summaries")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default=5)
    
    question_parser = subparsers.add_parser("question", help="Provide question for summaries")
    question_parser.add_argument("question", type=str, help="Search query for RAG")
    question_parser.add_argument("--limit", type=int, default=5)

    


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_command(query)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("\nRAG Response: ")
            print(results["llm_response"])
            
        case "summarize":
            query = args.query
            limit = args.limit
            results = summarize_command(query, limit)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("\nLLM Summary Response: ")
            print(results["llm_response"])
        
        case "citations":
            query = args.query
            limit = args.limit
            results = citations_command(query, limit)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("\nLLM Summary Response: ")
            print(results["llm_response"])

        case "question":
            
            question = args.question
            limit = args.limit
            results = question_command(question, limit)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("\nLLM Summary Response: ")
            print(results["llm_response"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()