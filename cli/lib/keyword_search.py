import string
from nltk.stem import PorterStemmer
from .utils import load_movies

stemmer = PorterStemmer()


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


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
    stemmed_tokens = []
    for token in filtered_words:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def tokenize_single_term(term):
    tokenized_terms = tokenize_text(term)
    if len(tokenized_terms) != 1:
        raise Exception("invalid number of terms")
    return tokenized_terms[0]

def load_stop_words():
    with open("data/stopwords.txt") as file:
        stop_words = file.read()
        stop_words = stop_words.splitlines()
        preprocessed_stopwords = [
            preprocess_text(stop_word) for stop_word in stop_words
        ]
        return preprocessed_stopwords

    

STOP_WORDS = load_stop_words()
