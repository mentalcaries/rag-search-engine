import argparse
import json
import string


def main() -> None:
	parser = argparse.ArgumentParser(description="Keyword Search CLI")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")

	search_parser = subparsers.add_parser("search", help="Search movies using keywords")
	search_parser.add_argument("query", type=str, help="Search query")

	args = parser.parse_args()

	match args.command:
		case "search":
			search_query = args.query
			query_tokens = tokenize_text(search_query)
			print(f"Searching for: {search_query}")
			movies = load_movies()
			for index, movie in enumerate(movies, 1):
				tokenized_title = tokenize_text(movie["title"])
				if has_matching_tokens(query_tokens, tokenized_title):
					print(f"{index}. {movie['title']}")
			pass
		case _:
			parser.print_help()

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
	for query_token in query_tokens:
		for title_token in title_tokens:
			if query_token in title_token:
				return True
	return False

def load_movies():
	with open("data/movies.json", "r") as file:
		data = json.load(file)
	return data["movies"]


def preprocess_text(input):
	input = input.lower()
	input = input.translate(str.maketrans("", "", string.punctuation))
	return input


def tokenize_text(text):
	text = preprocess_text(text)
	tokens = text.split()
	valid_tokens = []
	for token in tokens:
		if token:
			valid_tokens.append(token)
	filtered_words = []
	for word in valid_tokens:
		if word not in STOP_WORDS:
			filtered_words.append(word)
	return filtered_words

def load_stop_words():
	with open("data/stopwords.txt") as file:
		stop_words = file.read()
		stop_words = stop_words.splitlines()
		preprocessed_stopwords = [preprocess_text(stop_word) for stop_word in stop_words]
		return preprocessed_stopwords

STOP_WORDS = load_stop_words()

if __name__ == "__main__":
	main()

