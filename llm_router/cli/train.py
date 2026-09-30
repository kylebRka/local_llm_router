"""Train and calibrate the router with subject-family holdout evaluation."""

import argparse
import pickle
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CLASSIFIER_PATH, RoutingSettings
from ..settings import AppSettings
from ..training.dataset import all_records

texts = [row["text"] for row in all_records]
labels = [row["label"] for row in all_records]
families = [row["family"] for row in all_records]
from ..features import get_routing_features


def make_classifier():
    return CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), method="sigmoid", cv=5
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Router training / Обучение роутера")
    parser.add_argument("--lang", choices=("ru", "en"))
    args = parser.parse_args(argv)
    lang = args.lang or AppSettings.load().language
    if not (len(texts) == len(labels) == len(families)):
        raise ValueError("Dataset columns must have equal length")

    print(f"Creating embeddings for {len(texts)} requests..." if lang == "en" else f"Создаём embeddings для {len(texts)} запросов...")
    embeddings = get_routing_features(texts)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(embeddings, labels, families))
    train_embeddings, test_embeddings = embeddings[train_indices], embeddings[test_indices]
    train_labels = [labels[index] for index in train_indices]
    test_labels = [labels[index] for index in test_indices]
    print(f"Training: {dict(Counter(train_labels))}; holdout: {dict(Counter(test_labels))}" if lang == "en" else f"Обучение: {dict(Counter(train_labels))}; проверка: {dict(Counter(test_labels))}")
    print("Holdout subject families were not used for training." if lang == "en" else "Сюжеты проверочной части не встречались при обучении.")

    evaluated = make_classifier()
    evaluated.fit(train_embeddings, train_labels)
    predictions = evaluated.predict(test_embeddings)
    print(f"Holdout accuracy: {accuracy_score(test_labels, predictions):.2%}" if lang == "en" else f"Accuracy на отложенных сюжетах: {accuracy_score(test_labels, predictions):.2%}")
    print(classification_report(test_labels, predictions))
    probabilities = evaluated.predict_proba(test_embeddings)
    confidences = probabilities.max(axis=1)
    threshold = RoutingSettings().confidence_threshold
    covered = confidences >= threshold
    if covered.any():
        covered_correct = sum(pred == truth for pred, truth, keep in zip(predictions, test_labels, covered) if keep)
        print(f"At confidence >= {threshold:.0%}: {covered.sum()} of {len(test_labels)} requests; accuracy {covered_correct / covered.sum():.2%}" if lang == "en" else f"При confidence >= {threshold:.0%}: {covered.sum()} из {len(test_labels)} запросов; точность {covered_correct / covered.sum():.2%}")
    else:
        print(f"No holdout requests reached confidence >= {threshold:.0%}." if lang == "en" else f"На проверке нет запросов с confidence >= {threshold:.0%}.")

    print(f"Training final classifier on all {len(texts)} requests..." if lang == "en" else f"Обучаем итоговый классификатор на всех {len(texts)} запросах...")
    classifier = make_classifier()
    classifier.fit(embeddings, labels)
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLASSIFIER_PATH.open("wb") as file:
        pickle.dump(classifier, file)
    print(f"Classifier saved: {CLASSIFIER_PATH}" if lang == "en" else f"Классификатор сохранён: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
