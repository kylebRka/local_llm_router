"""Sentence embeddings, loaded only when classification is requested."""

from functools import lru_cache
from .config import EMBEDDING_PATH


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(EMBEDDING_PATH))


def get_embeddings(texts):
    return _model().encode(
        ["query: " + text for text in texts], normalize_embeddings=True
    )
