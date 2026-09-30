"""Bilingual category labels used by the console router."""

CATEGORY_NAMES = {
    "ru": {
        "general": "Общий вопрос", "programming": "Программирование",
        "mathematics": "Математика", "translation": "Перевод",
        "writing": "Работа с текстом", "databases": "Базы данных",
        "systems": "Системы", "science": "Наука",
        "algorithms": "Алгоритмы", "analysis": "Анализ",
    },
    "en": {
        "general": "General", "programming": "Programming",
        "mathematics": "Mathematics", "translation": "Translation",
        "writing": "Writing", "databases": "Databases",
        "systems": "Systems", "science": "Science",
        "algorithms": "Algorithms", "analysis": "Analysis",
    },
}

def category_name(language, category):
    return CATEGORY_NAMES.get(language, CATEGORY_NAMES["ru"]).get(category, category)
