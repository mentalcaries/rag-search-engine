import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    multimodal_parser = subparsers.add_parser("image_search")
    multimodal_parser.add_argument('image_path')

    verify_parser  = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image", type=str)
    args = parser.parse_args()

    if args.command == "image_search":
        results = image_search_command(args.image_path)
        for index, result in enumerate(results, start=1):
            print(f"{index}. {result['title']} (similarity: {result['score']:.3f})")
            print(f"    {result['description'][:100]}")

    else:
        image = args.image
        image_embedding = verify_image_embedding(image)
        print(image_embedding)




if __name__ == "__main__":
    main()
