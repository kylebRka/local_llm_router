"""Train a candidate router from reviewed real labels; promote only after evaluation."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ..config import CLASSIFIER_PATH, MODEL_ORDER, PROJECT_ROOT
from ..training.dataset import all_records as synthetic_records
from ..event_store import EventStore, classifier_hash, utc_now
from ..features import get_routing_features
from .real_data import approved_unique
from .train import make_classifier
from ..settings import AppSettings

LANG = "ru"

def L(ru, en):
    return en if LANG == "en" else ru

CANDIDATES = PROJECT_ROOT / 'classifier' / 'candidates'
ARCHIVE = PROJECT_ROOT / 'classifier' / 'archive'
MIN_PER_CLASS = 20
REAL_WEIGHT = 5.0


def normalized(text):
    return ' '.join(text.casefold().split())


def available_examples(store):
    examples = approved_unique(store)
    synthetic_texts = {normalized(row['text']) for row in synthetic_records}
    fresh = [row for row in examples if normalized(row['text']) not in synthetic_texts]
    return fresh, len(examples) - len(fresh)


def readiness(store, minimum=MIN_PER_CLASS):
    examples, overlap = available_examples(store)
    counts = Counter(row['label'] for row in examples)
    missing = {label: max(0, minimum - counts[label]) for label in MODEL_ORDER}
    return examples, counts, missing, overlap


def status(store, minimum=MIN_PER_CLASS):
    examples, counts, missing, overlap = readiness(store, minimum)
    print(L('Подтверждённых уникальных реальных запросов:', 'Unique approved real requests:'), len(examples))
    print(L('Совпали с синтетическим набором и исключены:', 'Excluded overlaps with synthetic data:'), overlap)
    print(L('По моделям:', 'By model:'), {label: counts[label] for label in MODEL_ORDER})
    print(L('Не хватает до порога:', 'Missing to threshold:'), missing)
    print(L('Статус:', 'Status:'), L('можно обучать кандидата','candidate training is ready') if not any(missing.values()) else L('пока недостаточно данных','not enough data yet'))
    return not any(missing.values())


def evaluate(model, x, labels):
    predicted = model.predict(x)
    return {
        'accuracy': float(accuracy_score(labels, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
        'macro_f1': float(f1_score(labels, predicted, labels=MODEL_ORDER, average='macro', zero_division=0)),
        'per_class': {
            label: {
                'count': int(sum(actual == label for actual in labels)),
                'correct': int(sum(actual == label and prediction == label
                                   for actual, prediction in zip(labels, predicted))),
            } for label in MODEL_ORDER
        },
    }


def train_candidate(store, minimum=MIN_PER_CLASS):
    real, counts, missing, overlap = readiness(store, minimum)
    if any(missing.values()):
        raise ValueError(f'Недостаточно подтверждённых примеров; не хватает: {missing}')
    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f'Действующий классификатор не найден: {CLASSIFIER_PATH}')

    train_real, test_real = train_test_split(
        real, test_size=0.20, stratify=[row['label'] for row in real], random_state=42,
    )
    real_texts = {normalized(row['text']) for row in real}
    synthetic = [row for row in synthetic_records if normalized(row['text']) not in real_texts]
    train_rows = [{'text': row['text'], 'label': row['label']} for row in synthetic] + train_real
    train_texts = [row['text'] for row in train_rows]
    train_labels = [row['label'] for row in train_rows]
    test_texts = [row['text'] for row in test_real]
    test_labels = [row['label'] for row in test_real]
    print(L(f'Признаки: {len(train_rows)} обучающих, {len(test_real)} отложенных реальных запросов...', f'Features: {len(train_rows)} training and {len(test_real)} held-out real requests...'))
    x_train = get_routing_features(train_texts)
    x_test = get_routing_features(test_texts)
    weights = np.array([1.0] * len(synthetic) + [REAL_WEIGHT] * len(train_real))

    with CLASSIFIER_PATH.open('rb') as file:
        baseline = pickle.load(file)  # This project's own trusted model artifact.
    baseline_metrics = evaluate(baseline, x_test, test_labels)
    candidate = make_classifier()
    candidate.fit(x_train, train_labels, sample_weight=weights)
    candidate_metrics = evaluate(candidate, x_test, test_labels)
    eligible = (
        candidate_metrics['macro_f1'] > baseline_metrics['macro_f1']
        and candidate_metrics['accuracy'] >= baseline_metrics['accuracy']
    )
    report = {
        'created_at_utc': utc_now(),
        'baseline_classifier_sha256': classifier_hash(),
        'minimum_per_class': minimum,
        'real_weight': REAL_WEIGHT,
        'real_count_by_class': {label: counts[label] for label in MODEL_ORDER},
        'real_overlap_with_synthetic': overlap,
        'real_train_request_ids': [row['request_id'] for row in train_real],
        'real_test_request_ids': [row['request_id'] for row in test_real],
        'synthetic_train_count': len(synthetic),
        'baseline': baseline_metrics,
        'candidate': candidate_metrics,
        'eligible_for_promotion': eligible,
    }
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    candidate_path = CANDIDATES / f'router_{stamp}.pkl'
    with candidate_path.open('wb') as file:
        pickle.dump(candidate, file)
    report['candidate_sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    report_path = candidate_path.with_suffix('.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with store.connection() as db:
        run_id = db.execute('''
            INSERT INTO training_runs (
                created_at, status, candidate_path, metrics_json,
                real_train_count, real_test_count, synthetic_count
            ) VALUES (?, 'candidate', ?, ?, ?, ?, ?)''',
            (report['created_at_utc'], str(candidate_path),
             json.dumps(report, ensure_ascii=False), len(train_real), len(test_real), len(synthetic)),
        ).lastrowid
    print(L(f'Кандидат #{run_id}: {candidate_path}', f'Candidate #{run_id}: {candidate_path}'))
    print(L('Действующий роутер:', 'Active router:'), baseline_metrics)
    print(L('Новый кандидат:', 'New candidate:'), candidate_metrics)
    print(L('Допуск к активации:', 'Eligible for promotion:'), 'да' if eligible else 'нет')
    return run_id, report


def promote(store, run_id):
    store.initialize()
    with store.connection() as db:
        row = db.execute('SELECT * FROM training_runs WHERE id=?', (run_id,)).fetchone()
    if row is None:
        raise ValueError(f'Кандидат #{run_id} не найден')
    if row['status'] != 'candidate':
        raise ValueError('Кандидат уже активирован или отклонён')
    report = json.loads(row['metrics_json'])
    if not report['eligible_for_promotion']:
        raise ValueError('Кандидат не прошёл сравнение с действующим роутером')
    if report['baseline_classifier_sha256'] != classifier_hash():
        raise ValueError('Действующий роутер изменился после обучения кандидата')
    candidate_path = Path(row['candidate_path'])
    if not candidate_path.is_file() or candidate_path.parent.resolve() != CANDIDATES.resolve():
        raise ValueError('Файл кандидата не найден в ожидаемой папке')
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != report['candidate_sha256']:
        raise ValueError('Файл кандидата изменился после оценки')
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backup = ARCHIVE / f'router_before_{run_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    shutil.copy2(CLASSIFIER_PATH, backup)
    replacement = CLASSIFIER_PATH.with_suffix('.pkl.pending')
    shutil.copy2(candidate_path, replacement)
    os.replace(replacement, CLASSIFIER_PATH)
    try:
        with store.connection() as db:
            db.execute("UPDATE training_runs SET status='promoted', promoted_at=? WHERE id=?",
                       (utc_now(), run_id))
    except Exception:
        shutil.copy2(backup, replacement)
        os.replace(replacement, CLASSIFIER_PATH)
        raise
    print(L(f'Кандидат #{run_id} активирован. Резервная копия: {backup}', f'Candidate #{run_id} promoted. Backup: {backup}'))
    print(L('Перезапустите работающий router.py, чтобы он загрузил новый классификатор.', 'Restart the running router.py to load the new classifier.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Переобучение / Retrain from reviewed real data')
    parser.add_argument('--db', type=Path, default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    training = sub.add_parser('train')
    training.add_argument('--min-per-class', type=int, default=MIN_PER_CLASS)
    activation = sub.add_parser('promote')
    activation.add_argument('run_id', type=int)
    parser.add_argument('--lang', choices=('ru','en'), help='Language / Язык')
    args = parser.parse_args(argv)
    global LANG
    LANG = args.lang or AppSettings.load().language
    store = EventStore(args.db) if args.db else EventStore()
    try:
        if args.command == 'status': status(store)
        elif args.command == 'train':
            if args.min_per_class < 10:
                raise ValueError('Минимум для обучения — 10 подтверждённых запросов на модель')
            train_candidate(store, args.min_per_class)
        elif args.command == 'promote': promote(store, args.run_id)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
