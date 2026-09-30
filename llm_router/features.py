"""Embedding plus a conservative lexical signal for requests to produce code."""

import re

import numpy as np

from .embedding import get_embeddings

CODE_CREATION = re.compile(
    r"\b(напиши|написать|сделай|сделать|создай|создать|нужен|нужна|реализуй|реализовать|"
    r"покажи|показать|сгенерируй|запрограммируй)\b.*"
    r"\b(python|питон\w*|код\w*|функци\w*|программ\w*|скрипт\w*|sql|калькулятор\w*)\b",
    re.IGNORECASE,
)


CODE_CREATION_EN = re.compile(r"\b(write|create|build|implement|generate|show|give)\b.*\b(python|code|function|script|program|calculator|sql)\b", re.IGNORECASE)

def get_routing_features(texts):
    embeddings = get_embeddings(texts)
    code_creation = np.array(
        [[10.0 if CODE_CREATION.search(text) or CODE_CREATION_EN.search(text) else 0.0] for text in texts],
        dtype=embeddings.dtype,
    )
    return np.hstack((embeddings, code_creation))
