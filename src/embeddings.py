from gensim.models import Word2Vec
import numpy as np


def train_word2vec(sentences, vector_size=100):
    """
    Train Word2Vec model for class label embeddings.
    """
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    return model


def get_embedding(model, word):
    """
    Retrieve embedding vector for a given class label.
    """
    return model.wv[word]


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
