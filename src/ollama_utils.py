from __future__ import annotations

import logging
import os
from urllib.parse import urljoin

import requests


def resolve_ollama_base_url(explicit_url: str | None = None) -> str:
    if explicit_url:
        return explicit_url
    if os.environ.get("OLLAMA_BASE_URL"):
        return os.environ["OLLAMA_BASE_URL"]
    host = os.environ.get("OLLAMA_HOST")
    if host:
        if host.startswith(("http://", "https://")):
            return host
        return f"http://{host}"
    if os.environ.get("RUNNING_IN_DOCKER") == "1":
        return "http://host.docker.internal:11434"
    return "http://127.0.0.1:11434"


def check_ollama(base_url: str, model: str, timeout: int = 4) -> bool:
    logging.info("Kontroluji Ollama: %s", base_url)
    try:
        response = requests.get(urljoin(base_url.rstrip("/") + "/", "api/tags"), timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logging.info("Ollama nedostupná, používám fallback")
        return False

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = {str(item.get("name", "")).split(":")[0] for item in models if isinstance(item, dict)}
    full_model_names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
    if model in full_model_names or model.split(":")[0] in model_names:
        logging.info("Ollama dostupná: model %s", model)
    else:
        logging.info("Ollama dostupná, model %s zatím není v seznamu /api/tags.", model)
    return True
