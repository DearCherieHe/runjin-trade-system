import json
import urllib.error
import urllib.request


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


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
