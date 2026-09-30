import contextlib
from contextlib import closing
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from llm_router.event_store import EventStore
from router import handle_question
from real_data import approve, approved_unique, export, reject
from retrain import readiness, status as retrain_status, train_candidate
from llm_router.routing import RouteDecision


def decision(label='E2B'):
    return RouteDecision(label, f'test/{label}', 'general', 2, .8, .6,
                         {'E2B': .8, 'E4B': .15, '12B': .05}, 'classifier', label)


class EventTrainingTests(unittest.TestCase):
    def test_real_events_feedback_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / 'events.sqlite3')
            request_id, decision_id = store.record_request('Реальный вопрос', decision(), 'Причина')
            response_id = store.record_response(
                decision_id,
                response={'output': [{'type': 'message', 'content': 'Ответ'}],
                          'stats': {'input_tokens': 3, 'total_output_tokens': 4,
                                    'tokens_per_second': 15}},
                answer='Ответ', timings={'total': 1.5, 'routing': .2},
            )
            store.record_feedback(response_id, 1, propose_label=True)
            with store.connection() as db:
                row = db.execute('''SELECT q.text, d.selected_model_label, r.text,
                                         m.input_tokens, m.total_seconds, l.model_label, l.status
                                  FROM requests q JOIN routing_decisions d ON d.request_id=q.id
                                  JOIN responses r ON r.decision_id=d.id
                                  JOIN metrics m ON m.response_id=r.id
                                  JOIN training_labels l ON l.request_id=q.id''').fetchone()
                self.assertEqual(tuple(row), ('Реальный вопрос', 'E2B', 'Ответ', 3, 1.5, 'E2B', 'pending'))
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
            approve(store, request_id, 'E2B')
            self.assertEqual(approved_unique(store)[0]['label'], 'E2B')
            output = Path(directory) / 'real.jsonl'
            export(store, output)
            self.assertEqual(json.loads(output.read_text())['text'], 'Реальный вопрос')
            self.assertFalse(retrain_status(store))
            with self.assertRaises(ValueError):
                train_candidate(store)
            reject(store, request_id)
            self.assertEqual(approved_unique(store), [])

    def test_no_automatic_label_and_error_response(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / 'events.sqlite3')
            _, decision_id = store.record_request('Другой вопрос', decision(), 'Причина')
            response_id = store.record_response(decision_id, response={'stats': {}}, answer='Ответ')
            store.record_feedback(response_id, -1, expected_model='E4B', propose_label=False)
            self.assertEqual(store.counts()['training_labels'], 0)
            _, failed_decision = store.record_request('Третий вопрос', decision(), 'Причина')
            failed_response = store.record_response(failed_decision, error=RuntimeError('offline'))
            with self.assertRaises(ValueError):
                store.record_feedback(failed_response, 1)

    def test_router_logs_success_failure_and_skips_benchmark(self):
        class Router:
            def route(self, question):
                return decision()
        class Explainer:
            def explain(self, route):
                return 'Простой общий вопрос.'
        class Manager:
            last_prepare_seconds = .1
            last_request_seconds = .2
            fail = False
            def chat(self, model, question):
                if self.fail:
                    raise RuntimeError('LM Studio offline')
                return {'output': [{'type': 'message', 'content': 'Готово'}],
                        'stats': {'input_tokens': 2, 'total_output_tokens': 3}}
            def extract_answer(self, result):
                return result['output'][0]['content']
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / 'events.sqlite3')
            manager = Manager()
            with patch('router.ask_feedback', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                handle_question('Реальный вопрос', Router(), Explainer(), manager,
                                event_store=store)
            self.assertEqual(store.counts()['requests'], 1)
            self.assertEqual(store.counts()['responses'], 1)
            manager.fail = True
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    handle_question('Ошибка', Router(), Explainer(), manager,
                                    event_store=store)
            with store.connection() as db:
                self.assertEqual(db.execute("SELECT status FROM responses ORDER BY id DESC LIMIT 1").fetchone()[0], 'error')
            with contextlib.redirect_stdout(io.StringIO()):
                manager.fail = False
                handle_question('Бенчмарк', Router(), Explainer(), manager,
                                benchmark_json=True, event_store=store)
            self.assertEqual(store.counts()['requests'], 2)

    def test_legacy_migration_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite3'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('''CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY, created_at TEXT, question TEXT, model_label TEXT,
                    category TEXT, difficulty INTEGER, confidence REAL, margin REAL,
                    probabilities_json TEXT, route_reason TEXT, explanation TEXT,
                    rating INTEGER, expected_model TEXT, comment TEXT)''')
                db.execute('''INSERT INTO feedback VALUES
                    (1, '2026-01-01', 'Рецепт', 'E4B', 'general', 2, .7, .4,
                     '{"E2B":0.2,"E4B":0.7,"12B":0.1}', 'classifier', 'Причина',
                     -1, 'E2B', '')''')
            store = EventStore(path)
            store.initialize()
            store.initialize()
            self.assertEqual(store.counts()['feedback'], 1)
            self.assertEqual(store.counts()['pending_labels'], 1)
            with store.connection() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM feedback_legacy').fetchone()[0], 1)
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])


class AlwaysE2B:
    def predict(self, features):
        import numpy as np
        return np.array(['E2B'] * len(features))


class LabelFromFeature:
    def fit(self, features, labels, sample_weight=None):
        self.weights_seen = sample_weight is not None
        return self

    def predict(self, features):
        import numpy as np
        labels = ('E2B', 'E4B', '12B')
        return np.array([labels[int(row[0])] for row in features])


class CandidatePipelineTests(unittest.TestCase):
    def test_candidate_train_evaluate_and_promote_with_fake_features(self):
        import hashlib
        import pickle
        import numpy as np
        from unittest.mock import patch
        import retrain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EventStore(root / 'events.sqlite3')
            store.initialize()
            with store.connection() as db:
                for label in ('E2B', 'E4B', '12B'):
                    for index in range(20):
                        request_id = db.execute(
                            "INSERT INTO requests(created_at, source, text) VALUES ('now','interactive',?)",
                            (f'{label} real {index}',),
                        ).lastrowid
                        db.execute("""INSERT INTO training_labels (
                            request_id, model_label, source, status, created_at, reviewed_at
                        ) VALUES (?, ?, 'manual', 'approved', 'now', 'now')""",
                        (request_id, label))
            active = root / 'router.pkl'
            with active.open('wb') as file:
                pickle.dump(AlwaysE2B(), file)
            def features(texts):
                order = {'E2B': 0, 'E4B': 1, '12B': 2}
                return np.array([[order[text.split()[0]]] for text in texts], dtype=float)
            synthetic = [{'text': f'{label} synthetic', 'label': label}
                         for label in ('E2B', 'E4B', '12B')]
            with patch.object(retrain, 'CLASSIFIER_PATH', active), \
                 patch.object(retrain, 'CANDIDATES', root / 'candidates'), \
                 patch.object(retrain, 'ARCHIVE', root / 'archive'), \
                 patch.object(retrain, 'synthetic_records', synthetic), \
                 patch.object(retrain, 'get_routing_features', features), \
                 patch.object(retrain, 'make_classifier', LabelFromFeature), \
                 patch.object(retrain, 'classifier_hash',
                              lambda: hashlib.sha256(active.read_bytes()).hexdigest()):
                run_id, report = train_candidate(store, minimum=20)
                self.assertTrue(report['eligible_for_promotion'])
                self.assertEqual(report['real_test_request_ids'].__len__(), 12)
                with active.open('rb') as file:
                    self.assertIsInstance(pickle.load(file), AlwaysE2B)
                retrain.promote(store, run_id)
                with active.open('rb') as file:
                    self.assertIsInstance(pickle.load(file), LabelFromFeature)
                self.assertEqual(len(list((root / 'archive').glob('*.pkl'))), 1)


if __name__ == '__main__':
    unittest.main()
