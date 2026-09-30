"""Router benchmark for any number of active requests, with full measurements."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from ..settings import AppSettings
from time import perf_counter

import psutil

ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ROOT / 'data' / 'benchmark' / 'requests.jsonl'
MARKER = 'ROUTER_BENCHMARK_JSON: '
FIELDS = [
    'id',
    'theme',
    'request',
    'status',
    'exit_code',
    'model',
    'model_id',
    'category',
    'difficulty',
    'confidence',
    'margin',
    'reason',
    'explanation',
    'input_tokens',
    'output_tokens',
    'reasoning_tokens',
    'ttft_seconds',
    'generation_tokens_per_second',
    'generation_seconds_estimate',
    'routing_seconds',
    'explanation_seconds',
    'model_prepare_seconds',
    'model_request_seconds',
    'lm_studio_total_seconds',
    'in_process_seconds',
    'wall_seconds',
    'router_peak_rss_mb',
    'router_peak_cpu_percent',
    'lm_studio_peak_rss_mb',
    'lm_studio_peak_cpu_percent',
    'system_peak_memory_used_mb',
    'system_lowest_memory_available_mb',
    'system_peak_swap_used_mb',
    'system_peak_cpu_percent',
    'error',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def mb(number):
    return round(number / 1024**2, 2)


def load_requests(path=REQUESTS):
    """Read JSONL; allow one or more commented-out /* ... */ blocks."""
    records = []
    in_block = False
    for number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if in_block:
            if '*/' in line:
                in_block = False
            continue
        if line.startswith('/*'):
            in_block = '*/' not in line
            continue
        if line.startswith('#') or line.startswith('//'):
            continue
        if line.startswith('*/'):
            raise ValueError(f'Лишнее закрытие комментария, строка {number}')
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Некорректный JSON, строка {number}: {exc.msg}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'Ожидался JSON-объект, строка {number}')
        records.append(row)
    if in_block:
        raise ValueError('Незакрытый блок комментария /* ... */')
    if not records:
        raise ValueError('Нет активных запросов в файле')
    ids = [row.get('id') for row in records]
    texts = [row.get('text') for row in records]
    if any(type(value) is not int or value <= 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('У каждого запроса должен быть уникальный положительный целый id')
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError('У каждого запроса должен быть непустой text')
    if len({value.strip().casefold() for value in texts}) != len(texts):
        raise ValueError('Тексты активных запросов должны быть уникальны')
    if any(not isinstance(row.get('theme'), str) or not row['theme'].strip() for row in records):
        raise ValueError('У каждого запроса должна быть непустая theme')
    return records


def parse_payload(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    raise ValueError('Роутер не вернул структурированные метрики')


class ResourceMonitor:
    """Sample process trees and system statistics while a request is running."""

    def __init__(self, pid, interval=0.5):
        self.pid, self.interval = pid, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.process_cache = {}
        self.peaks = {
            k: None
            for k in FIELDS
            if k.startswith(
                ('router_peak_', 'lm_studio_peak_', 'system_peak_', 'system_lowest_')
            )
        }

    def start(self):
        psutil.cpu_percent(interval=None)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)
        return {
            **self.peaks,
            'resource_samples': self.samples,
            'sample_interval_seconds': self.interval,
        }

    def update(self, key, value, minimum=False):
        if value is None:
            return
        old = self.peaks[key]
        self.peaks[key] = (
            value if old is None else (min(old, value) if minimum else max(old, value))
        )

    def totals(self, processes):
        rss = cpu = 0.0
        found = False
        for process in processes:
            try:
                cached = self.process_cache.get(process.pid)
                if cached is None or cached.create_time() != process.create_time():
                    cached = process
                    self.process_cache[process.pid] = cached
                rss += cached.memory_info().rss
                cpu += cached.cpu_percent(interval=None)
                found = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return (mb(rss), round(cpu, 2)) if found else (None, None)

    def sample(self):
        try:
            process = psutil.Process(self.pid)
            router = [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            router = []
        rss, cpu = self.totals(router)
        self.update('router_peak_rss_mb', rss)
        self.update('router_peak_cpu_percent', cpu)
        studio = []
        for process in psutil.process_iter(['name', 'exe']):
            try:
                identity = (
                    f"{process.info['name'] or ''} {process.info['exe'] or ''}".lower()
                )
                if 'lm studio' in identity or 'lmstudio' in identity:
                    studio.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        rss, cpu = self.totals(studio)
        self.update('lm_studio_peak_rss_mb', rss)
        self.update('lm_studio_peak_cpu_percent', cpu)
        memory = psutil.virtual_memory()
        self.update('system_peak_memory_used_mb', mb(memory.used))
        self.update(
            'system_lowest_memory_available_mb', mb(memory.available), minimum=True
        )
        self.update('system_peak_swap_used_mb', mb(psutil.swap_memory().used))
        self.update('system_peak_cpu_percent', psutil.cpu_percent(interval=None))
        self.samples += 1

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except (OSError, RuntimeError):
                pass
            self.stop_event.wait(self.interval)


def summarize(record):
    payload = record.get('payload') or {}
    decision = payload.get('decision') or {}
    timings = payload.get('timings_seconds') or {}
    stats = (payload.get('lm_studio_response') or {}).get('stats') or {}
    resources = record.get('resources') or {}
    output = stats.get('total_output_tokens')
    tps = stats.get('tokens_per_second')
    estimate = (
        output / tps
        if isinstance(output, (int, float))
        and isinstance(tps, (int, float))
        and tps > 0
        else None
    )
    row = dict(
        id=record['id'],
        theme=record['theme'],
        request=record['request'],
        status=record['status'],
        exit_code=record['exit_code'],
        model=decision.get('model_label'),
        model_id=decision.get('model_id'),
        category=decision.get('category'),
        difficulty=decision.get('difficulty'),
        confidence=decision.get('confidence'),
        margin=decision.get('margin'),
        reason=decision.get('reason'),
        explanation=payload.get('explanation'),
        input_tokens=stats.get('input_tokens'),
        output_tokens=output,
        reasoning_tokens=stats.get('reasoning_output_tokens'),
        ttft_seconds=stats.get('time_to_first_token_seconds'),
        generation_tokens_per_second=tps,
        generation_seconds_estimate=estimate,
        routing_seconds=timings.get('routing'),
        explanation_seconds=timings.get('explanation'),
        model_prepare_seconds=timings.get('model_prepare'),
        model_request_seconds=timings.get('model_request'),
        lm_studio_total_seconds=timings.get('lm_studio_total'),
        in_process_seconds=timings.get('total_in_process'),
        wall_seconds=record['wall_seconds'],
        error=record.get('error'),
    )
    row.update(
        {
            key: resources.get(key)
            for key in FIELDS
            if key.startswith(
                ('router_peak_', 'lm_studio_peak_', 'system_peak_', 'system_lowest_')
            )
        }
    )
    return row


def run_request(item, timeout, language=None):
    started_at, started = now(), perf_counter()
    stdout = stderr = ''
    code = error = payload = None
    monitor = None
    resources = {}
    try:
        process = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'router.py'), '--benchmark-json'] + (['--lang', language] if language else []),
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        monitor = ResourceMonitor(process.pid)
        monitor.start()
        try:
            stdout, stderr = process.communicate(item['text'] + '\n', timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            error = f'Тайм-аут после {timeout} с'
        code = process.returncode
        if error is None and code != 0:
            error = f'Роутер завершился с кодом {code}'
        if error is None:
            try:
                payload = parse_payload(stdout)
            except (ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
    except (OSError, ValueError) as exc:
        error = f'Ошибка запуска: {exc}'
    finally:
        if monitor is not None:
            resources = monitor.stop()
    return dict(
        id=item['id'],
        theme=item['theme'],
        request=item['text'],
        started_at_utc=started_at,
        finished_at_utc=now(),
        wall_seconds=perf_counter() - started,
        status='ok' if error is None else 'error',
        exit_code=code,
        error=error,
        stdout=stdout,
        stderr=stderr,
        payload=payload,
        resources=resources,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Бенчмарк запросов / Local router benchmark'
    )
    parser.add_argument(
        '--timeout', type=int, default=900, help='Секунд на запрос / Seconds per request'
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmark_runs')
    parser.add_argument('--requests', type=Path, default=REQUESTS,
                        help='JSONL с запросами / Request JSONL (default: data/benchmark/requests.jsonl)')
    parser.add_argument('--limit', type=int, help='Первые N запросов / First N active requests')
    parser.add_argument('--check', action='store_true', help='Проверить список без моделей / Check requests without models')
    parser.add_argument('--lang', choices=('ru', 'en'), help='Language / Язык')
    args = parser.parse_args(argv)
    language = args.lang or AppSettings.load().language
    if args.timeout <= 0:
        parser.error('--timeout должен быть положительным')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit должен быть положительным')
    requests = load_requests(args.requests)
    if args.limit is not None:
        requests = requests[:args.limit]
    total = len(requests)
    if args.check:
        print(f'Active requests: {total}' if language == 'en' else f'Активных запросов: {total}')
        for position, item in enumerate(requests, 1):
            print(f"{position:3d}. id={item['id']} [{item['theme']}] {item['text']}")
        return
    snapshot = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in requests)
    run_dir = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'requests.jsonl').write_text(snapshot, encoding='utf-8')
    info = dict(
        created_at_utc=now(),
        request_count=len(requests),
        requests_sha256=hashlib.sha256(snapshot.encode('utf-8')).hexdigest(),
        source_requests_path=str(args.requests.resolve()),
        python=sys.version,
        platform=platform.platform(),
        cpu_logical_count=psutil.cpu_count(logical=True),
        cpu_physical_count=psutil.cpu_count(logical=False),
        system_total_ram_mb=mb(psutil.virtual_memory().total),
        psutil_version=psutil.__version__,
        timeout_seconds_per_request=args.timeout,
        monitor_note='CPU/RAM sampled periodically; system load includes other apps. '
        'LM Studio process access may be denied. Generation duration is estimated as output tokens / tokens per second.',
    )
    (run_dir / 'run_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Results: {run_dir}' if language == 'en' else f'Результаты: {run_dir}', flush=True)
    with (run_dir / 'results.jsonl').open('w', encoding='utf-8') as json_file, (
        run_dir / 'summary.csv'
    ).open('w', newline='', encoding='utf-8-sig') as csv_file, (
        run_dir / 'full_output.txt'
    ).open(
        'w', encoding='utf-8'
    ) as text_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for position, item in enumerate(requests, 1):
            print(f"[{position:03d}/{total}] {item['text']}", flush=True)
            record = run_request(item, args.timeout, language)
            json_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            json_file.flush()
            writer.writerow(summarize(record))
            csv_file.flush()
            text_file.write(
                f"\n{'='*80}\n{position:03d}/{total} (id={item['id']}) | {item['theme']} | {item['text']}\n"
                f"status={record['status']} | wall={record['wall_seconds']:.3f}s | exit={record['exit_code']} | error={record['error']}\n"
                f"--- STDOUT ---\n{record['stdout']}\n--- STDERR ---\n{record['stderr']}\n"
            )
            text_file.flush()
            print(f"  {record['status']} — {record['wall_seconds']:.1f} s" if language == 'en' else f"  {record['status']} — {record['wall_seconds']:.1f} с", flush=True)
    print(f'Done. Files: {run_dir}' if language == 'en' else f'Готово. Файлы в {run_dir}')


if __name__ == '__main__':
    main()
