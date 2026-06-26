from __future__ import annotations

from collections.abc import Mapping

ENV_ALLOWLIST = {
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
}

SECRET_MARKERS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AWS_SECRET",
    "OPENAI",
    "ANTHROPIC",
    "HF_TOKEN",
)

REDACTED_VALUE = "[REDACTED]"


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def redact_environment(
    environ: Mapping[str, str],
    *,
    allowlist: set[str] | None = None,
    include_redacted_secret_keys: bool = True,
) -> dict[str, str]:
    allowed = allowlist or ENV_ALLOWLIST
    result: dict[str, str] = {}
    for key in sorted(environ):
        value = environ[key]
        if is_secret_key(key):
            if include_redacted_secret_keys:
                result[key] = REDACTED_VALUE
            continue
        if key in allowed:
            result[key] = value
    return result
