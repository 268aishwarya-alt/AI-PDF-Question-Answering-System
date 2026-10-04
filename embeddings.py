from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):

        print("Loading semantic embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def create_embeddings(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )