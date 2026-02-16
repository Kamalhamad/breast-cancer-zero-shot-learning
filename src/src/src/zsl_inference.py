import numpy as np
from embeddings import cosine_similarity


def predict_unseen(feature_vector, classifier, encoder, class_embeddings):
    """
    Predict unseen class using embedding similarity.
    """
    predicted_label_index = classifier.predict([feature_vector])[0]
    predicted_label = encoder.inverse_transform([predicted_label_index])[0]

    best_match = None
    best_score = -1

    for class_name, embedding in class_embeddings.items():
        score = cosine_similarity(feature_vector, embedding)
        if score > best_score:
            best_score = score
            best_match = class_name

    return best_match
