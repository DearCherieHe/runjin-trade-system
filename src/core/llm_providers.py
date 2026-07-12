from dataclasses import dataclass


@dataclass
class LLMProvider:
    key: str
    name: str
    default_model: str
    local: bool
    enabled: bool
    notes: str


PROVIDERS = {
    "ollama": LLMProvider(
        key="ollama",
        name="Ollama",
        default_model="qwen2.5:7b",
        local=True,
        enabled=True,
        notes="本地投研问答，适合草稿、反方证据、检查清单。",
    ),
    "openai_compatible": LLMProvider(
        key="openai_compatible",
        name="OpenAI-compatible",
        default_model="configured-by-env",
        local=False,
        enabled=False,
        notes="预留给 OpenAI 兼容网关，不在本地默认启用。",
    ),
}


TASK_MODEL_POLICY = {
    "thesis_review": {"provider": "ollama", "model": "qwen2.5:7b", "reason": "中文投研归纳，本地低成本。"},
    "risk_check": {"provider": "ollama", "model": "qwen2.5:7b", "reason": "反方证据和失效条件检查。"},
    "report_draft": {"provider": "ollama", "model": "qwen2.5:7b", "reason": "报告初稿可本地生成后人工校订。"},
}


def provider_table():
    return [
        {
            "key": provider.key,
            "name": provider.name,
            "default_model": provider.default_model,
            "local": provider.local,
            "enabled": provider.enabled,
            "notes": provider.notes,
        }
        for provider in PROVIDERS.values()
    ]


def task_policy_table():
    return [
        {"task": task, **policy}
        for task, policy in TASK_MODEL_POLICY.items()
    ]
