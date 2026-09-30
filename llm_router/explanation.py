"""Short, locally generated explanations grounded in route metadata."""

import re
from functools import lru_cache

from .config import EXPLAINER_PATH, RoutingSettings

TOPIC_DESCRIPTIONS = {
    "general": (
        "простой общий вопрос", "общий вопрос средней сложности", "сложный общий вопрос"
    ),
    "programming": (
        "простая задача по программированию",
        "задача по программированию средней сложности",
        "сложная задача по программированию",
    ),
    "mathematics": (
        "простая математическая задача",
        "математическая задача средней сложности",
        "сложная математическая задача",
    ),
    "translation": (
        "простой запрос на перевод", "запрос на перевод средней сложности",
        "сложный запрос на перевод",
    ),
    "writing": (
        "простая задача по работе с текстом",
        "задача по работе с текстом средней сложности",
        "сложная задача по работе с текстом",
    ),
    "databases": (
        "простой вопрос о базах данных", "задача по базам данных средней сложности",
        "сложная задача по базам данных",
    ),
    "systems": (
        "простой вопрос о работе систем", "задача по устройству систем средней сложности",
        "сложная задача по устройству систем",
    ),
    "science": (
        "простой научный вопрос", "научный вопрос средней сложности",
        "сложный научный вопрос",
    ),
    "algorithms": (
        "простой вопрос об алгоритмах", "задача по алгоритмам средней сложности",
        "сложная задача по алгоритмам",
    ),
    "analysis": (
        "простой аналитический вопрос",
        "аналитический вопрос средней сложности",
        "сложный аналитический вопрос",
    ),
}

TOPIC_DESCRIPTIONS_EN = {
    "general": ("a simple general question", "a moderately complex general question", "a complex general question"),
    "programming": ("a simple programming task", "a moderately complex programming task", "a complex programming task"),
    "mathematics": ("a simple math question", "a moderately complex math problem", "a complex math problem"),
    "translation": ("a short translation request", "a moderately complex translation request", "a complex translation request"),
    "writing": ("a simple writing task", "a moderately complex writing task", "a complex writing task"),
    "databases": ("a simple database question", "a moderately complex database task", "a complex database task"),
    "systems": ("a simple systems question", "a moderately complex systems task", "a complex systems task"),
    "science": ("a simple science question", "a moderately complex science question", "a complex science question"),
    "algorithms": ("a simple algorithms question", "a moderately complex algorithms task", "a complex algorithms task"),
    "analysis": ("a simple analysis request", "a moderately complex analysis request", "a complex analysis request"),
}


@lru_cache(maxsize=1)
def _load_model():
    from mlx_lm import load

    if not EXPLAINER_PATH.is_dir():
        raise FileNotFoundError(f"Модель объяснений не найдена: {EXPLAINER_PATH}")
    return load(str(EXPLAINER_PATH))


def describe_request(category: str, difficulty: int, language: str = "ru") -> str:
    """Turn category and the 0–9 difficulty estimate into a short task description."""
    if not 0 <= difficulty <= 9:
        raise ValueError("difficulty must be between 0 and 9")
    level = 0 if difficulty <= 2 else 1 if difficulty <= 5 else 2
    topics = TOPIC_DESCRIPTIONS_EN if language == "en" else TOPIC_DESCRIPTIONS
    return topics.get(category, topics["general"])[level]


def reason_clause(decision, settings: RoutingSettings, language: str = "ru") -> str:
    """Produce the facts the small model may phrase, without inventing a cause."""
    topic = describe_request(decision.category, decision.difficulty, language)
    if language == "en":
        return {
            "classifier": f"this is {topic}",
            "low_confidence": f"this is {topic} and low confidence calls for a stronger model",
            "low_confidence_max_tier": f"this is {topic} and no stronger model is available",
            "hard_request_borderline": f"this is {topic} and borderline confidence calls for a stronger model",
            "reuse_loaded_e4b": f"this is {topic} and E4B is already loaded for a short translation",
            "manual_override": f"this is {topic} and the user selected this model",
        }[decision.reason]
    if decision.reason == "manual_override":
        return f"это {topic}, и эту модель выбрал пользователь"
    if decision.reason == "classifier":
        return f"это {topic}"
    if decision.reason == "low_confidence":
        return f"это {topic}, а низкая уверенность потребовала более мощной модели"
    if decision.reason == "low_confidence_max_tier":
        return f"это {topic}, а при низкой уверенности более мощной модели нет"
    if decision.reason == "hard_request_borderline":
        return f"это {topic}, а пограничная уверенность потребовала более мощной модели"
    if decision.reason == "reuse_loaded_e4b":
        return f"это {topic}, а E4B уже загружена и позволяет избежать переключения моделей"
    raise ValueError(f"Неизвестная причина маршрута: {decision.reason}")


def canonical_explanation(decision, settings: RoutingSettings, language: str = "ru") -> str:
    clause = reason_clause(decision, settings, language)
    return (f"Model {decision.model_label} was selected because {clause}." if language == "en"
            else f"Выбрана модель {decision.model_label}, потому что {clause}.")


def _validate_generated(text: str, decision, settings: RoutingSettings, language: str = "ru") -> str | None:
    if not isinstance(text, str):
        return None
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*", ", ", text)
    if not text.endswith("."):
        text += "."
    if len(text.split()) > 30 or text.count(".") != 1 or any(c in text for c in "!?"):
        return None
    if language == "en":
        if not re.match(rf"^Model {re.escape(decision.model_label)} was selected because ", text, re.IGNORECASE):
            return None
        if describe_request(decision.category, decision.difficulty, language) not in text.lower():
            return None
        if any(label in text for label in ("E2B", "E4B", "12B") if label != decision.model_label):
            return None
        return text if all(word in text.lower() for word in ({"manual_override": ("user", "selected"), "low_confidence": ("confidence", "stronger"), "reuse_loaded_e4b": ("already loaded",)}.get(decision.reason, ())) ) else None
    if not re.match(
        rf"^Выбрана модель {re.escape(decision.model_label)}\s*,?\s*потому что\s+",
        text, re.IGNORECASE,
    ):
        return None
    if any(label in text for label in ("E2B", "E4B", "12B") if label != decision.model_label):
        return None
    lower = text.lower()
    if describe_request(decision.category, decision.difficulty) not in lower:
        return None
    if decision.reason == "low_confidence":
        if "уверенност" not in lower or "мощн" not in lower:
            return None
    elif decision.reason == "low_confidence_max_tier":
        if "уверенност" not in lower or "нет" not in lower:
            return None
    elif decision.reason == "hard_request_borderline":
        if "пограничн" not in lower or "мощн" not in lower:
            return None
    elif decision.reason == "reuse_loaded_e4b":
        if "уже загруж" not in lower or "переключ" not in lower:
            return None
    if decision.reason == "manual_override" and "пользовател" not in lower:
        return None
    return text


class ExplanationGenerator:
    def __init__(self, settings=None, model_loader=None, generate_text=None, language="ru"):
        self.settings = settings if settings is not None else RoutingSettings()
        self.model_loader = model_loader if model_loader is not None else _load_model
        self.generate_text = generate_text
        self.language = language

    def explain(self, decision) -> str:
        fallback = canonical_explanation(decision, self.settings, self.language)
        topic = describe_request(decision.category, decision.difficulty, self.language)
        if self.language == "en":
            messages = [
                {"role": "system", "content": f"Write exactly one short English sentence starting: Model {decision.model_label} was selected because. Use the given topic and reason; no field names."},
                {"role": "user", "content": f"Topic: {decision.category}; difficulty: {decision.difficulty}/9; task: {topic}; reason: {reason_clause(decision, self.settings, self.language)}."},
            ]
        else:
            messages = None
        user_facts = (
            f"Категория {decision.category}; сложность {decision.difficulty}/9; "
            f"описание запроса: {topic}."
        )
        if decision.reason == "low_confidence":
            user_facts += " Низкая уверенность потребовала более мощной модели; укажи это после описания."
        elif decision.reason == "low_confidence_max_tier":
            user_facts += " Уверенность ниже порога, но более мощной модели нет; укажи это после описания."
        elif decision.reason == "hard_request_borderline":
            user_facts += " Пограничная уверенность и высокая сложность потребовали более мощной модели; укажи это после описания."
        elif decision.reason == "reuse_loaded_e4b":
            user_facts += " Меньшая модель была рекомендована, но E4B уже загружена; короткий перевод быстрее выполнить без переключения моделей."
        elif decision.reason == "manual_override":
            user_facts += " Модель выбрана пользователем вручную; укажи это после описания."
        messages = messages or [
            {
                "role": "system",
                "content": (
                    f"Ответь только коротким предложением по форме «Выбрана модель "
                    f"{decision.model_label}, потому что это ...». Не выводи названия полей."
                ),
            },
            {"role": "user", "content": user_facts},
        ]
        try:
            model, tokenizer = self.model_loader()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            if self.generate_text is None:
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler

                response = generate(
                    model, tokenizer, prompt=prompt, max_tokens=56,
                    verbose=False, sampler=make_sampler(temp=0),
                )
            else:
                response = self.generate_text(model, tokenizer, prompt)
        except (ImportError, OSError, RuntimeError, ValueError):
            return fallback
        return _validate_generated(response, decision, self.settings, self.language) or fallback
