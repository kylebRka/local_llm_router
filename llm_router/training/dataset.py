"""Load and validate the Russian and English training datasets."""

import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "training" / "requests_ru.jsonl"

with DATA_PATH.open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

texts = [record["text"] for record in records]
labels = [record["label"] for record in records]
categories = [record["category"] for record in records]
difficulties = [record["difficulty"] for record in records]
families = [record["family"] for record in records]

if Counter(labels) != {"E2B": 500, "E4B": 500, "12B": 500}:
    raise ValueError("Dataset must contain exactly 500 requests per model")
if len(set(text.casefold().strip() for text in texts)) != len(texts):
    raise ValueError("Dataset contains duplicate requests")

EN_DATA_PATH = DATA_PATH.with_name("requests_en.jsonl")
with EN_DATA_PATH.open(encoding="utf-8") as file:
    english_records = [json.loads(line) for line in file if line.strip()]
if Counter(row["label"] for row in english_records) != {"E2B": 500, "E4B": 500, "12B": 500}:
    raise ValueError("English dataset must contain exactly 500 requests per model")
if len({row["text"].casefold().strip() for row in english_records}) != len(english_records):
    raise ValueError("English dataset contains duplicate requests")
all_records = records + english_records
