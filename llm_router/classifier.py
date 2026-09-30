"""Adapter around the trained model selector."""

import pickle
from functools import lru_cache

from .config import CLASSIFIER_PATH


@lru_cache(maxsize=1)
def load_classifier():
    # The pickle is a local, trusted training artifact. Never load an arbitrary file.
    with CLASSIFIER_PATH.open("rb") as file:
        return pickle.load(file)


class ModelClassifier:
    def __init__(self, model=None, embed=None):
        self._model = model
        self._embed = embed

    def predict_probabilities(self, text):
        from .features import get_routing_features

        model = self._model if self._model is not None else load_classifier()
        embed = self._embed if self._embed is not None else get_routing_features
        probabilities = model.predict_proba(embed([text]))[0]
        return dict(zip(map(str, model.classes_), map(float, probabilities)))


def predict(text):
    """Compatibility helper for callers of the previous API."""
    probabilities = ModelClassifier().predict_probabilities(text)
    prediction = max(probabilities, key=probabilities.get)
    return prediction, probabilities[prediction], probabilities
