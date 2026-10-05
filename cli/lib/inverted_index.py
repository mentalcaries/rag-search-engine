import pickle
import os
from .keyword_search import tokenize_text, load_movies, tokenize_single_term
from collections import Counter
import math
from .utils import BM25_K1


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

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

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
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

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1) -> float:
        raw_tf = self.get_tf(doc_id, term)
        bm25_saturated = (raw_tf * (k1 + 1)) / (raw_tf + k1)
        return bm25_saturated


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


def bm25tf_command(doc_id: int, term: str, k1: int) -> float:
    index.load()
    term = tokenize_single_term(term)
    bm25tf = index.get_bm25_tf(doc_id, term, BM25_K1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
