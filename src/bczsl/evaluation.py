from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(y_true, y_pred):
    """
    Evaluate classification performance.
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return acc, cm, report
