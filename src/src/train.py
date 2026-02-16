import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def train_classifier(features, labels):
    """
    Train logistic regression classifier on seen classes.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, encoded_labels)

    return clf, encoder
