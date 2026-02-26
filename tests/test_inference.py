import numpy as np

from bczsl.zsl_inference import predict_unseen


class DummyClassifier:
    def predict(self, X):
        return [0]


class DummyEncoder:
    def inverse_transform(self, y):
        return ["seen_class"]


def test_predict_unseen_selects_best_embedding_match():
    feature = np.array([1.0, 0.0])
    class_embeddings = {
        "u1": np.array([0.0, 1.0]),
        "u2": np.array([1.0, 0.0]),
    }

    predicted = predict_unseen(
        feature, DummyClassifier(), DummyEncoder(), class_embeddings
    )
    assert predicted == "u2"
