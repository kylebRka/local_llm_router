# Benchmark / Бенчмарк

## Measured run / Проведённый прогон

On 29 September 2026, the console benchmark processed the [100 published Russian requests](../data/benchmark/requests_100.jsonl) on a local Apple Silicon Mac with 16 GB of RAM, using LM Studio and the E2B, E4B and 12B model slots. The input file SHA-256 was `7c6bf22b213e8268a02f424ab9c7f91822a282b6fa65d1c5f2d6bb381b2c963a`. Each request started a fresh Python process and generated a complete answer. All 100 runs returned successfully.

29 сентября 2026 года консольный бенчмарк обработал [100 опубликованных русскоязычных запросов](../data/benchmark/requests_100.jsonl) на локальном Mac с Apple Silicon и 16 ГБ ОЗУ через LM Studio и модели уровней E2B, E4B и 12B. Каждый запрос запускался в отдельном процессе Python и генерировал полный ответ. Все 100 запусков завершились успешно.

| Measurement / Метрика | Result / Результат |
| --- | ---: |
| Completed requests / Успешных запросов | 100 / 100 |
| Selected E2B / E4B / 12B | 32 / 25 / 43 |
| Median end-to-end wall time / Медиана полного времени | 96.03 s |
| 90th percentile wall time / 90-й процентиль | 190.89 s |
| Median routing time / Медиана маршрутизации | 6.14 s |
| Median explanation time / Медиана причины выбора | 1.00 s |

The median wall times for the requests routed to E2B, E4B and 12B were 36.93 s, 82.37 s and 174.12 s respectively. **These are not a controlled model-speed comparison:** each tier received different requests, outputs had different lengths, model loading varied, and total wall time includes process startup. System resource sampling also included other applications. The benchmark recorded latency and routing decisions, **not answer quality or whether the chosen tier was optimal**. These figures describe this one local run and should not be generalized to other machines or models.

Медиана полного времени для запросов, направленных к E2B, E4B и 12B, составила 36,93 с, 82,37 с и 174,12 с соответственно. **Это не контролируемое сравнение скорости моделей:** уровни получили разные запросы, длина ответов различалась, время загрузки моделей менялось, а полное время включает запуск процесса. В измерение системных ресурсов попадала нагрузка других приложений. Бенчмарк фиксировал время и выбор маршрута, **но не оценивал качество ответов или оптимальность выбора модели**.

The original full responses and process telemetry remain local. To reproduce the workflow with your own installed models, run:

```sh
.venv/bin/python benchmark.py --lang ru --check --requests data/benchmark/requests_100.jsonl
.venv/bin/python benchmark.py --lang ru --requests data/benchmark/requests_100.jsonl
```

This historical run predates this documentation and CI update. Its exact dependency versions were not recorded, so the figures are not guaranteed to reproduce even with the same input file. Other model versions or hardware can also produce different numbers.

Этот прогон был выполнен до обновления документации и CI. Точные версии библиотек во время прогона не зафиксированы, поэтому даже с тем же файлом запросов результаты могут отличаться. На других версиях моделей и другом оборудовании показатели также будут иными.
