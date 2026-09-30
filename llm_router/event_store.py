"""SQLite storage for real router events and reviewed training labels."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import CLASSIFIER_PATH, FEEDBACK_PATH, MODEL_IDS, MODEL_ORDER

SCHEMA_PATH = Path(__file__).resolve().parent / 'storage' / 'schema.sql'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def classifier_hash():
    try:
        return hashlib.sha256(CLASSIFIER_PATH.read_bytes()).hexdigest()
    except OSError:
        return None


class EventStore:
    def __init__(self, path=FEEDBACK_PATH):
        self.path = Path(path)

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA busy_timeout = 10000')
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        """Create schema; migrate old feedback transactionally without deleting it."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            names = {row['name'] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            legacy = False
            if 'feedback' in names:
                columns = {row['name'] for row in db.execute('PRAGMA table_info(feedback)')}
                if 'question' in columns and 'response_id' not in columns:
                    if 'feedback_legacy' in names:
                        raise RuntimeError('Обнаружены две несовместимые старые таблицы feedback')
                    db.execute('ALTER TABLE feedback RENAME TO feedback_legacy')
                    legacy = True
            for statement in SCHEMA_PATH.read_text(encoding='utf-8').split(';'):
                if statement.strip():
                    db.execute(statement)
            if legacy:
                for row in db.execute('SELECT * FROM feedback_legacy ORDER BY id').fetchall():
                    request_id = db.execute(
                        'INSERT INTO requests(created_at, source, text) VALUES (?, ?, ?)',
                        (row['created_at'], 'legacy', row['question']),
                    ).lastrowid
                    decision_id = db.execute('''
                        INSERT INTO routing_decisions (
                            request_id, created_at, predicted_model_label, selected_model_label,
                            selected_model_id, category, difficulty, confidence, margin,
                            probabilities_json, route_reason, explanation, classifier_sha256
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        (request_id, row['created_at'], row['model_label'], row['model_label'],
                         MODEL_IDS[row['model_label']], row['category'], row['difficulty'],
                         row['confidence'], row['margin'], row['probabilities_json'],
                         row['route_reason'], row['explanation'], None),
                    ).lastrowid
                    response_id = db.execute('''
                        INSERT INTO responses(decision_id, created_at, status)
                        VALUES (?, ?, 'legacy_unknown')''',
                        (decision_id, row['created_at']),
                    ).lastrowid
                    feedback_id = db.execute('''
                        INSERT INTO feedback(response_id, created_at, rating, expected_model_label, comment)
                        VALUES (?,?,?,?,?)''',
                        (response_id, row['created_at'], row['rating'], row['expected_model'], row['comment']),
                    ).lastrowid
                    if row['rating'] == -1 and row['expected_model'] in MODEL_ORDER:
                        db.execute('''
                            INSERT INTO training_labels (
                                request_id, model_label, source, source_feedback_id, status, created_at
                            ) VALUES (?,?, 'feedback', ?, 'pending', ?)''',
                            (request_id, row['expected_model'], feedback_id, row['created_at']),
                        )

    def record_request(self, question, decision, explanation):
        self.initialize()
        with self.connection() as db:
            created_at = utc_now()
            request_id = db.execute(
                'INSERT INTO requests(created_at, source, text) VALUES (?, ?, ?)',
                (created_at, 'interactive', question),
            ).lastrowid
            decision_id = db.execute('''
                INSERT INTO routing_decisions (
                    request_id, created_at, predicted_model_label, selected_model_label,
                    selected_model_id, category, difficulty, confidence, margin,
                    probabilities_json, route_reason, explanation, classifier_sha256
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (request_id, created_at, decision.predicted_label or decision.model_label,
                 decision.model_label, decision.model_id, decision.category, decision.difficulty,
                 decision.confidence, decision.margin,
                 json.dumps(decision.probabilities, ensure_ascii=False), decision.reason,
                 explanation, classifier_hash()),
            ).lastrowid
            return request_id, decision_id

    def record_response(self, decision_id, response=None, answer=None, error=None, timings=None):
        if response is None and error is None:
            raise ValueError('response or error is required')
        timings = timings or {}
        stats = response.get('stats', {}) if response is not None else {}
        with self.connection() as db:
            now = utc_now()
            response_id = db.execute('''
                INSERT INTO responses(decision_id, created_at, status, text, error_message, raw_json)
                VALUES (?,?,?,?,?,?)''',
                (decision_id, now, 'error' if error else 'success', answer, str(error) if error else None,
                 json.dumps(response, ensure_ascii=False, default=str) if response is not None else None),
            ).lastrowid
            db.execute('''
                INSERT INTO metrics (
                    response_id, created_at, total_seconds, routing_seconds, explanation_seconds,
                    model_prepare_seconds, model_request_seconds, input_tokens, output_tokens,
                    reasoning_tokens, ttft_seconds, tokens_per_second, raw_stats_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (response_id, now, timings.get('total'), timings.get('routing'),
                 timings.get('explanation'), timings.get('model_prepare'),
                 timings.get('model_request'), stats.get('input_tokens'),
                 stats.get('total_output_tokens'), stats.get('reasoning_output_tokens'),
                 stats.get('time_to_first_token_seconds'), stats.get('tokens_per_second'),
                 json.dumps(stats, ensure_ascii=False, default=str)),
            )
            return response_id

    def record_feedback(self, response_id, rating, expected_model=None, comment='', propose_label=False):
        if rating not in (-1, 1):
            raise ValueError('rating must be -1 or 1')
        if expected_model is not None and expected_model not in MODEL_ORDER:
            raise ValueError('Unknown expected model')
        with self.connection() as db:
            response = db.execute('''
                SELECT r.status, d.request_id FROM responses r
                JOIN routing_decisions d ON d.id = r.decision_id
                WHERE r.id = ?''', (response_id,)).fetchone()
            if response is None:
                raise ValueError('Unknown response_id')
            if response['status'] == 'error':
                raise ValueError('Cannot rate a failed response')
            now = utc_now()
            feedback_id = db.execute('''
                INSERT INTO feedback(response_id, created_at, rating, expected_model_label, comment)
                VALUES (?,?,?,?,?)''',
                (response_id, now, rating, expected_model, comment),
            ).lastrowid
            proposed_model = expected_model if rating == -1 else (
                db.execute('''SELECT selected_model_label FROM routing_decisions d
                              JOIN responses r ON r.decision_id = d.id
                              WHERE r.id = ?''', (response_id,)).fetchone()[0] if propose_label else None
            )
            if propose_label and proposed_model is not None:
                db.execute('''
                    INSERT INTO training_labels (
                        request_id, model_label, source, source_feedback_id, status, created_at
                    ) VALUES (?,?, 'feedback', ?, 'pending', ?)
                    ON CONFLICT(request_id) DO NOTHING''',
                    (response['request_id'], proposed_model, feedback_id, now),
                )
            return feedback_id

    def counts(self):
        self.initialize()
        with self.connection() as db:
            tables = ('requests', 'responses', 'feedback', 'training_labels')
            counts = {table: db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in tables}
            counts['approved_labels'] = db.execute(
                "SELECT COUNT(*) FROM training_labels WHERE status='approved'"
            ).fetchone()[0]
            counts['pending_labels'] = db.execute(
                "SELECT COUNT(*) FROM training_labels WHERE status='pending'"
            ).fetchone()[0]
            return counts

    def approved_examples(self):
        self.initialize()
        with self.connection() as db:
            return [dict(row) for row in db.execute('''
                SELECT r.id AS request_id, r.text, l.model_label, l.source, l.reviewed_at
                FROM training_labels l JOIN requests r ON r.id = l.request_id
                WHERE l.status = 'approved' AND r.source IN ('interactive','legacy')
                ORDER BY r.id''')]
