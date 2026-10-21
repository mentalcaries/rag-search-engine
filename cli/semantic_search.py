import numpy as np
from sentence_transformers import SentenceTransformer
import os
from lib.utils import (
    load_movies,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
)
import re
import json


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embeddings(self, text):
        if not text.strip():
            raise ValueError("text cannot be empty)")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movie_texts = []
        for document in documents:
            self.document_map[document["id"]] = document
            doc_text = f"{document['title']}: {document['description']}"
            movie_texts.append(doc_text)
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embeddings(query)
        scores = []
        for i, doc in enumerate(self.documents):
            cos_score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((cos_score, doc))
        sorted_results = sorted(scores, key=lambda item: item[0], reverse=True)
        top_results = []
        for result in sorted_results[:limit]:
            score, movie = result
            top_results.append(
                {
                    "score": score,
                    "title": movie["title"],
                    "description": movie["description"],
                }
            )
        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for doc_index, document in enumerate(documents):
            if document["description"] == "":
                continue
            self.document_map[document["id"]] = document
            semantic_chunks = semantic_chunk(document["description"], 4, 1)
            chunks.extend(semantic_chunks)
            for index, chunk in enumerate(semantic_chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": doc_index,
                        "chunk_idx": index,
                        "total_chunks": len(semantic_chunks),
                    }
                )
        self.embeddings = self.model.encode(chunks, show_progress_bar=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
            )
        return self.embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embeddings(query)
        chunk_scores = []
        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": index,
                    "movie_idx": self.chunk_metadata[index]["movie_idx"],
                    "score": similarity,
                }
            )
        movie_index_scores = {}
        for chunk_score in chunk_scores:
            if (
                chunk_score["movie_idx"] not in movie_index_scores
                or chunk_score["score"] > movie_index_scores[chunk_score["movie_idx"]]
            ):
                movie_index_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        sorted_movie_scores = sorted(
            movie_index_scores.items(), key=lambda item: item[1], reverse=True
        )

        top_movies_by_limit = sorted_movie_scores[:limit]
        results = []
        for movie_idx, score in top_movies_by_limit:
            doc = self.documents[movie_idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 3),
                }
            )
        return results


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embeddings(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query, limit=5):
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['title']} (score: {result['score']})\n{result['description'][:100]}..."
        )


def chunk_command(text: str, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        if i > 0:
            chunks.append(" ".join(words[i - overlap : i + chunk_size]))
        else:
            chunks.append(" ".join(words[i : i + chunk_size]))

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def semantic_chunk(text: str, max_chunk_size=4, overlap=0):
    text = text.strip()
    if text == "":
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        return [text]

    chunks = []
    step = max_chunk_size - overlap
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        chunk_sentences = [sentence.strip() for sentence in chunk_sentences]
        chunk_sentences = [sentence for sentence in chunk_sentences if sentence]

        chunk = " ".join(chunk_sentences)

        if chunk:
            chunks.append(chunk)

        i += step

    return chunks


def semantic_chunk_command(text: str, max_chunk_size=4, overlap=0):
    chunks = semantic_chunk(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def embed_chunks_command():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query, limit=5):
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    results = chunked_semantic_search.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
