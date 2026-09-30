"""Persistent user settings, independent of the training artifacts."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .config import MODEL_IDS, PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / 'config' / 'settings.json'


@dataclass
class AppSettings:
    language: str = 'ru'
    lm_studio_url: str = 'http://127.0.0.1:1234'
    model_ids: dict[str, str] = field(default_factory=lambda: dict(MODEL_IDS))
    reuse_loaded_e4b: bool = True
    collect_feedback_labels: bool = False

    def validate(self):
        if self.language not in ('ru', 'en'):
            raise ValueError('Language must be ru or en')
        parsed = urlparse(self.lm_studio_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError('LM Studio address must be an HTTP(S) origin, e.g. http://192.168.1.10:1234')
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError('Invalid LM Studio port') from exc
        if set(self.model_ids) != set(MODEL_IDS) or any(not isinstance(v, str) or not v.strip() for v in self.model_ids.values()):
            raise ValueError('Set a full model name for E2B, E4B and 12B')
        return self

    def save(self, path=SETTINGS_PATH):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temp.replace(path)

    @classmethod
    def load(cls, path=SETTINGS_PATH):
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding='utf-8'))
        valid = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid}
        filtered['model_ids'] = {**MODEL_IDS, **filtered.get('model_ids', {})}
        return cls(**filtered).validate()
