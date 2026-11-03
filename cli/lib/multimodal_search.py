from PIL import Image
from sentence_transformers import SentenceTransformer
from semantic_search import cosine_similarity
from lib.utils import load_movies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", docs=[]):
        self.model = SentenceTransformer(model_name_or_path=model_name)
        self.docs = docs
        self.texts = self.generate_texts()
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        
    def generate_texts(self):
        doc_texts = []
        for doc in self.docs:
            doc_texts.append(f"{doc['title']}: {doc['description']}")
        return doc_texts

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        similarities = []
        for text_embedding in self.text_embeddings:
            similarities.append(cosine_similarity(image_embedding, text_embedding))

        combined = list(zip(self.docs, similarities))
        sorted_results = sorted(combined, key=lambda item: item[1], reverse=True)
        results = []

        for doc, score in sorted_results:
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score": score,
            })

        return results[:5]


def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    search = MultimodalSearch(docs=movies)
    return search.search_with_image(image_path)
