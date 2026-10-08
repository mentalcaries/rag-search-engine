import json

CACHE_DIR = "cache"
BM25_K1 = 1.5
BM25_B = 0.75
MOVIE_EMBEDDINGS_PATH = "cache/movie_embeddings.npy"

def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)
    return data["movies"]
