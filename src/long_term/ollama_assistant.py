import json
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "ollama.yaml"


def load_ollama_config():
    config = {"base_url": DEFAULT_OLLAMA_URL, "default_model": DEFAULT_OLLAMA_MODEL}
    if not CONFIG_PATH.exists():
        return config
    try:
        import yaml

        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        loaded = {}
    config.update({key: value for key, value in loaded.items() if value})
    return config


def save_ollama_config(base_url, default_model):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_url": base_url.rstrip("/"),
        "default_model": default_model.strip(),
    }
    try:
        import yaml

        CONFIG_PATH.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception:
        CONFIG_PATH.write_text(
            f"base_url: {payload['base_url']}\ndefault_model: {payload['default_model']}\n",
            encoding="utf-8",
        )
    return CONFIG_PATH


def _request_json(url, payload=None, timeout=60):
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_status(base_url=DEFAULT_OLLAMA_URL, timeout=3):
    try:
        payload = _request_json(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"available": False, "models": [], "message": str(exc)}
    models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
    return {"available": True, "models": models, "message": "Ollama is available"}


def ask_ollama(prompt, model=DEFAULT_OLLAMA_MODEL, base_url=DEFAULT_OLLAMA_URL, timeout=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }
    response = _request_json(f"{base_url.rstrip('/')}/api/generate", payload=payload, timeout=timeout)
    return response.get("response", "").strip()
