import numpy as np
from gensim.models import Word2Vec


def train_word2vec(sentences, vector_size=100):
    """Train Word2Vec model for class label embeddings."""
    return Word2Vec(sentences, vector_size=vector_size, min_count=1)


def get_embedding(model, word):
    """Retrieve embedding vector for a given class label."""
    return model.wv[word]


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)
