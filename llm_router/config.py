"""Runtime settings for local routing."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = PROJECT_ROOT / "classifier" / "router.pkl"
EMBEDDING_PATH = PROJECT_ROOT / "models" / "multilingual-e5-small"
EXPLAINER_PATH = PROJECT_ROOT / "models" / "qwen3-0.6b-4bit"
FEEDBACK_PATH = PROJECT_ROOT / "feedback" / "feedback.sqlite3"

MODEL_IDS = {
    "E2B": "google/gemma-4-e2b",
    "E4B": "google/gemma-4-e4b",
    "12B": "google/gemma-4-12b",
}
MODEL_ORDER = tuple(MODEL_IDS)


@dataclass(frozen=True)
class RoutingSettings:
    confidence_threshold: float = 0.60

    def __post_init__(self):
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
