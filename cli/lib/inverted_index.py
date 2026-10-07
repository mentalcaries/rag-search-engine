import pickle
import os
from .keyword_search import tokenize_text, load_movies, tokenize_single_term
from collections import Counter
import math
from .utils import CACHE_DIR, BM25_K1, BM25_B


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, float] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        total_tokens = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = total_tokens

    def __get_avg_doc_length(self) -> float:
        total_docs = len(self.docmap)
        if total_docs == 0:
            return 0.0
        total_doc_length = sum(self.doc_lengths.values())
        return total_doc_length / total_docs

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
            with open(self.doc_lengths_path, "rb") as file:
                self.doc_lengths = pickle.load(file)
        except FileNotFoundError as error:
            print(f"file not found: {error.filename}")

    def get_tf(self, doc_id, term) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        doc_frequency = len(self.get_documents(term))
        bm25 = math.log((total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        doc_length_normalization = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * doc_length_normalization)
        return bm25_tf

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query: str, limit=5):
        tokenized_query = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            for token in tokenized_query:
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += self.bm25(doc_id, token)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores[:limit]


index = InvertedIndex()


def build_command() -> None:
    index.build()
    index.save()


def calc_tf(doc_id: int, term: str):
    tokenized_term = tokenize_single_term(term)
    frequency = index.get_tf(doc_id, tokenized_term)
    return frequency


def calculate_idf(term: str):
    tokenized_term = tokenize_single_term(term)
    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(tokenized_term))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf


def tf_command(doc_id: int, term: str) -> None:
    index.load()
    frequency = calc_tf(doc_id, term)
    print(frequency)


def idf_command(term: str):
    index.load()
    idf = calculate_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str) -> None:
    index.load()
    idf = calculate_idf(term)
    tf = calc_tf(doc_id, term)
    tf_idf = idf * tf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term: str) -> float:
    index.load()
    term = tokenize_single_term(term)
    bm25_idf = index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")


def bm25tf_command(doc_id: int, term: str, k1: int, b: float) -> float:
    index.load()
    term = tokenize_single_term(term)
    bm25tf = index.get_bm25_tf(doc_id, term, BM25_K1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

def bm25search_command(query: str, limit: int):
    index.load()
    results = index.bm25_search(query, limit)
    for num, (id, score) in enumerate(results, start=1):
        movie_name = index.docmap[id]
        print(f"{num}. ({id} {movie_name['title']} - Score: {score:.2f})")
    return results
