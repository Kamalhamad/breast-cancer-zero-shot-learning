import numpy as np

from bczsl.embeddings import cosine_similarity


def test_cosine_similarity_identity():
    vec = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vec, vec) == 1.0


def test_cosine_similarity_zero_vector_guard():
    vec = np.array([0.0, 0.0, 0.0])
    other = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(vec, other) == 0.0
