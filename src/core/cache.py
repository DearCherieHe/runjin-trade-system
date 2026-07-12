import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"


def stable_key(namespace, payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}_{digest}"


class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key, ttl_seconds=None):
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if ttl_seconds is not None and time.time() - payload.get("created_at", 0) > ttl_seconds:
            return None
        return payload.get("value")

    def set(self, key, value):
        path = self._path(key)
        payload = {"created_at": time.time(), "value": value}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def list_entries(self):
        rows = []
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "key": path.stem,
                    "created_at": payload.get("created_at"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
        return rows

    def clear_namespace(self, namespace):
        deleted = 0
        for path in self.cache_dir.glob(f"{namespace}_*.json"):
            path.unlink()
            deleted += 1
        return deleted
