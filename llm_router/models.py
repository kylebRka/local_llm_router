import urllib.request
import urllib.error
import json
from time import perf_counter

from .config import MODEL_IDS

BASE_URL = "http://127.0.0.1:1234"
MODELS = MODEL_IDS  # Compatibility alias for existing callers.


class ModelManager:

    def __init__(self, base_url=None, language="ru", quiet=False):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.language = language
        self.quiet = quiet
        self.current_model = None
        self.current_instance_id = None
        self.last_prepare_seconds = None
        self.last_request_seconds = None

    def _status(self, message):
        if not self.quiet:
            print(message)

    # ==================================================
    # HTTP-запрос к LM Studio
    # ==================================================

    def _request(self, method, endpoint, data=None):

        url = self.base_url + endpoint

        body = None

        if data is not None:
            body = json.dumps(data).encode("utf-8")

        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method=method
        )

        try:

            with urllib.request.urlopen(request) as response:

                response_data = response.read().decode("utf-8")

                if not response_data:
                    return {}

                return json.loads(response_data)

        except urllib.error.HTTPError as error:

            error_body = error.read().decode("utf-8")

            raise RuntimeError(f"LM Studio API error " f"{error.code}: {error_body}")

        except urllib.error.URLError as error:

            raise RuntimeError(f"Could not connect to LM Studio: {error}" if self.language == "en" else f"Не удалось подключиться к LM Studio: {error}")

    # ==================================================
    # Получить список моделей
    # ==================================================

    def get_models(self):

        return self._request("GET", "/api/v1/models")

    # ==================================================
    # Получить загруженные модели
    # ==================================================

    def get_loaded_models(self):

        data = self.get_models()

        loaded = []

        for model in data.get("models", []):

            for instance in model.get("loaded_instances", []):

                loaded.append({"key": model["key"], "instance_id": instance["id"]})

        return loaded

    # ==================================================
    # Загрузка модели
    # ==================================================

    def load_model(self, model):

        self._status(f"Loading model: {model}" if self.language == "en" else f"Загружаю модель: {model}")

        result = self._request("POST", "/api/v1/models/load", {"model": model})

        self._status("Model loaded." if self.language == "en" else "Модель успешно загружена.")

        return result

    # ==================================================
    # Выгрузка модели
    # ==================================================

    def unload_model(self, instance_id):

        self._status(f"Unloading model: {instance_id}" if self.language == "en" else f"Выгружаю модель: {instance_id}")

        result = self._request(
            "POST", "/api/v1/models/unload", {"instance_id": instance_id}
        )

        self._status("Model unloaded." if self.language == "en" else "Модель выгружена.")

        return result

    # ==================================================
    # Убедиться, что нужная модель загружена
    # ==================================================

    def ensure_model_loaded(self, model):

        loaded_models = self.get_loaded_models()

        # ------------------------------------------------
        # Проверяем, загружена ли нужная модель
        # ------------------------------------------------

        for loaded in loaded_models:

            if loaded["key"] == model:

                self._status(f"Model already loaded: {model}" if self.language == "en" else f"Модель уже загружена: {model}")

                self.current_model = model

                self.current_instance_id = loaded["instance_id"]

                return

        # ------------------------------------------------
        # Если загружена другая модель —
        # выгружаем её
        # ------------------------------------------------

        for loaded in loaded_models:

            self._status(f"Another model is loaded: {loaded['key']}" if self.language == "en" else f"Найдена другая модель: {loaded['key']}")

            self.unload_model(loaded["instance_id"])

        # ------------------------------------------------
        # Загружаем нужную модель
        # ------------------------------------------------

        result = self.load_model(model)

        self.current_model = model

        # LM Studio может вернуть instance ID
        # в ответе загрузки.

        self.current_instance_id = (
            result.get("instance_id") or result.get("model_instance_id") or model
        )

    # ==================================================
    # Отправка сообщения модели
    # ==================================================

    def chat(self, model, question):

        started = perf_counter()
        self.ensure_model_loaded(model)
        prepared = perf_counter()
        self.last_prepare_seconds = prepared - started

        data = {"model": model, "input": question}

        self._status("Sending request to model..." if self.language == "en" else "Отправляю запрос в модель...")

        result = self._request("POST", "/api/v1/chat", data)
        self.last_request_seconds = perf_counter() - prepared

        return result

    # ==================================================
    # Получить текст ответа
    # ==================================================

    @staticmethod
    def extract_answer(result):

        output = result.get("output", [])

        for item in output:

            if item.get("type") == "message":

                return item.get("content", "")

        return ""
