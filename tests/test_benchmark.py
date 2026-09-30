import unittest

import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_requests_are_distinct_and_complete(self):
        rows = benchmark.load_requests()
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(len({row['id'] for row in rows}), len(rows))
        self.assertEqual(len({row['text'] for row in rows}), len(rows))

    def test_variable_count_and_commented_block(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.jsonl'
            path.write_text('/*\n' + json.dumps({'id': 1, 'theme': 'old', 'text': 'old'}) +
                            '\n*/\n' +
                            json.dumps({'id': 7, 'theme': 'one', 'text': 'first'}) + '\n' +
                            json.dumps({'id': 9, 'theme': 'two', 'text': 'second'}) + '\n',
                            encoding='utf-8')
            self.assertEqual([row['id'] for row in benchmark.load_requests(path)], [7, 9])

    def test_check_mode_does_not_run_models_or_create_results(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'runs'
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                benchmark.main(['--check', '--output-dir', str(output)])
            self.assertIn(f'Активных запросов: {len(benchmark.load_requests())}', stream.getvalue())
            self.assertFalse(output.exists())

    def test_parse_and_summarize_full_response(self):
        payload = {
            'decision': {'model_label': 'E2B', 'model_id': 'test/model',
                         'category': 'general', 'difficulty': 2,
                         'confidence': .72, 'margin': .5, 'reason': 'classifier'},
            'explanation': 'Выбрана модель E2B, потому что это простой общий вопрос.',
            'timings_seconds': {'routing': .1, 'explanation': .2,
                                'model_prepare': 1.0, 'model_request': 2.0,
                                'lm_studio_total': 3.0, 'total_in_process': 3.3},
            'lm_studio_response': {'output': [{'type': 'message', 'content': 'Рецепт'}],
                                   'stats': {'input_tokens': 10, 'total_output_tokens': 40,
                                             'reasoning_output_tokens': 0,
                                             'time_to_first_token_seconds': .3,
                                             'tokens_per_second': 20}},
        }
        import json
        stdout = 'Ответ: Рецепт\n' + benchmark.MARKER + json.dumps(payload, ensure_ascii=False)
        self.assertEqual(benchmark.parse_payload(stdout), payload)
        record = {'id': 1, 'theme': 'everyday', 'request': 'Борщ', 'status': 'ok',
                  'exit_code': 0, 'wall_seconds': 4.0, 'payload': payload,
                  'resources': {'router_peak_rss_mb': 500}, 'error': None}
        row = benchmark.summarize(record)
        self.assertEqual(row['model'], 'E2B')
        self.assertEqual(row['generation_seconds_estimate'], 2)
        self.assertEqual(row['router_peak_rss_mb'], 500)
        self.assertEqual(set(row), set(benchmark.FIELDS))


if __name__ == '__main__':
    unittest.main()
