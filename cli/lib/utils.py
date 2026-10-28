import json

CACHE_DIR = "cache"
BM25_K1 = 1.5
BM25_B = 0.75
MOVIE_EMBEDDINGS_PATH = "cache/movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"

def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)
    return data["movies"]

def load_dataset():
    with open("data/golden_dataset.json", "r") as file:
        data = json.load(file)
    return data["test_cases"]
