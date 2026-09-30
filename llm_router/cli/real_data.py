"""Inspect real events, review routing labels, and export an auditable dataset."""

import argparse
import json
from pathlib import Path

from ..config import MODEL_ORDER, PROJECT_ROOT
from ..event_store import EventStore, utc_now
from ..settings import AppSettings

LANG = "ru"

def L(ru, en):
    return en if LANG == "en" else ru

DEFAULT_EXPORT = PROJECT_ROOT / 'data' / 'training' / 'real_requests.jsonl'


def status(store):
    counts = store.counts()
    with store.connection() as db:
        by_label = dict(db.execute('''
            SELECT model_label, COUNT(*) FROM training_labels
            WHERE status='approved' GROUP BY model_label''').fetchall())
    print(L('События:', 'Events:'), counts['requests'], L('запросов,','requests,'), counts['responses'], L('ответов,','responses,'), counts['feedback'], L('отзывов','feedback entries'))
    print(L('Метки:','Labels:'), counts['pending_labels'], L('ожидают проверки,','pending,'), counts['approved_labels'], L('подтверждены','approved'))
    print(L('Подтверждено по моделям:', 'Approved by model:'), {label: by_label.get(label, 0) for label in MODEL_ORDER})
    print(L('Для переобучения нужно не менее 20 уникальных подтверждённых запросов на каждую модель.', 'Retraining requires at least 20 unique approved requests for each model.'))


def list_requests(store, state='pending', limit=20):
    store.initialize()
    with store.connection() as db:
        rows = db.execute('''
            SELECT q.id, q.created_at, q.source, q.text,
                   d.selected_model_label, f.rating, f.expected_model_label,
                   l.model_label, l.status
            FROM requests q
            JOIN routing_decisions d ON d.id = (
                SELECT MAX(id) FROM routing_decisions WHERE request_id=q.id
            )
            LEFT JOIN responses r ON r.decision_id=d.id
            LEFT JOIN feedback f ON f.response_id=r.id
            LEFT JOIN training_labels l ON l.request_id=q.id
            WHERE (?='all' OR (?='unlabeled' AND l.id IS NULL)
                   OR (? IN ('pending','approved','rejected') AND l.status=?))
            ORDER BY q.id DESC LIMIT ?''',
            (state, state, state, state, limit),
        ).fetchall()
    for row in rows:
        text = row['text'].replace('\n', ' ')
        print(f"#{row['id']} {row['source']} | {L('модель','model')} {row['selected_model_label']} | "
              f"{L('отзыв','feedback')} {row['rating']} | {L('предложено','proposed')} {row['model_label'] or '-'} "
              f"({row['status'] or '-'}) | {text[:110]}")
    if not rows:
        print(L('Запросов с таким состоянием нет.', 'No requests with this status.'))


def show_request(store, request_id):
    store.initialize()
    with store.connection() as db:
        row = db.execute('''
            SELECT q.id, q.created_at, q.source, q.text AS question,
                   d.predicted_model_label, d.selected_model_label, d.category,
                   d.difficulty, d.confidence, d.margin, d.route_reason, d.explanation,
                   r.status AS response_status, r.text AS answer, r.error_message,
                   f.rating, f.expected_model_label, f.comment,
                   l.model_label AS training_label, l.status AS label_status
            FROM requests q
            JOIN routing_decisions d ON d.id = (
                SELECT MAX(id) FROM routing_decisions WHERE request_id=q.id
            )
            LEFT JOIN responses r ON r.decision_id=d.id
            LEFT JOIN feedback f ON f.response_id=r.id
            LEFT JOIN training_labels l ON l.request_id=q.id
            WHERE q.id=?''', (request_id,)).fetchone()
    if row is None:
        raise ValueError(f'Запрос #{request_id} не найден')
    for key in row.keys():
        print(f'{key}: {row[key]}')


def approve(store, request_id, label):
    if label not in MODEL_ORDER:
        raise ValueError('Неизвестная модель')
    store.initialize()
    with store.connection() as db:
        request = db.execute('SELECT source FROM requests WHERE id=?', (request_id,)).fetchone()
        if request is None:
            raise ValueError(f'Запрос #{request_id} не найден')
        if request['source'] not in ('interactive', 'legacy'):
            raise ValueError('Нельзя размечать тестовый запрос')
        now = utc_now()
        db.execute('''
            INSERT INTO training_labels (
                request_id, model_label, source, status, created_at, reviewed_at
            ) VALUES (?, ?, 'manual', 'approved', ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                model_label=excluded.model_label, source='manual',
                status='approved', reviewed_at=excluded.reviewed_at''',
            (request_id, label, now, now),
        )
    print(L(f'Запрос #{request_id}: подтверждена метка {label}', f'Request #{request_id}: label {label} approved'))


def reject(store, request_id):
    store.initialize()
    with store.connection() as db:
        cursor = db.execute('''
            UPDATE training_labels SET status='rejected', reviewed_at=?
            WHERE request_id=?''', (utc_now(), request_id))
        if cursor.rowcount == 0:
            raise ValueError('Для этого запроса нет предложенной метки')
    print(L(f'Метка запроса #{request_id} отклонена', f'Request #{request_id} label rejected'))


def approved_unique(store):
    examples = store.approved_examples()
    unique = {}
    conflicts = []
    for row in examples:
        key = ' '.join(row['text'].casefold().split())
        previous = unique.get(key)
        if previous and previous['label'] != row['model_label']:
            conflicts.append((previous['request_id'], row['request_id']))
        elif previous is None:
            unique[key] = {'request_id': row['request_id'], 'text': row['text'],
                           'label': row['model_label'], 'source': row['source'],
                           'reviewed_at': row['reviewed_at']}
    if conflicts:
        raise ValueError(f'Противоречивые метки одинаковых запросов: {conflicts}')
    return list(unique.values())


def export(store, output=DEFAULT_EXPORT):
    examples = approved_unique(store)
    if not examples:
        print(L('Подтверждённых меток пока нет; файл не изменён.', 'No approved labels yet; file unchanged.'))
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in examples)
    output.write_text(payload, encoding='utf-8')
    print(L(f'Экспортировано {len(examples)} подтверждённых уникальных запросов: {output}', f'Exported {len(examples)} unique approved requests: {output}'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Просмотр и разметка запросов / Review real requests')
    parser.add_argument('--db', type=Path, default=None, help='Путь к SQLite / SQLite path')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    listing = sub.add_parser('list')
    listing.add_argument('--status', choices=('pending','approved','rejected','unlabeled','all'), default='pending')
    listing.add_argument('--limit', type=int, default=20)
    showing = sub.add_parser('show')
    showing.add_argument('request_id', type=int)
    labeling = sub.add_parser('approve')
    labeling.add_argument('request_id', type=int)
    labeling.add_argument('model_label', choices=MODEL_ORDER)
    rejecting = sub.add_parser('reject')
    rejecting.add_argument('request_id', type=int)
    exporting = sub.add_parser('export')
    exporting.add_argument('--output', type=Path, default=DEFAULT_EXPORT)
    parser.add_argument('--lang', choices=('ru','en'), help='Language / Язык')
    args = parser.parse_args(argv)
    global LANG
    LANG = args.lang or AppSettings.load().language
    store = EventStore(args.db) if args.db else EventStore()
    try:
        if args.command == 'status': status(store)
        elif args.command == 'list': list_requests(store, args.status, args.limit)
        elif args.command == 'show': show_request(store, args.request_id)
        elif args.command == 'approve': approve(store, args.request_id, args.model_label)
        elif args.command == 'reject': reject(store, args.request_id)
        elif args.command == 'export': export(store, args.output)
    except ValueError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
