from __future__ import annotations

import re
from typing import Any

from mcp.types import ToolAnnotations

# read only tools no need for user confirmation
READ_ONLY_ALLOWLIST = frozenset({"search_catalog", "get_book", "search_policies", "verify_member", "get_account"})

IDENTITY_TOOL = "verify_member"

# the model shouldnt see these arguments
HIDDEN_PARAMS = frozenset({"member_id", "idempotency_key"})
MAX_FAILED_VERIFICATIONS = 3
MAX_TOOL_RESULT_CHARS = 6000

_YES = {
    "yes",
    "y",
    "yes please",
    "yep",
    "yeah",
    "confirm",
    "confirmed",
    "ok",
    "okay",
    "go ahead",
    "do it",
    "sure",
    "please do",
    "oui",
}
_NO = {"no", "n", "nope", "cancel", "stop", "don't", "do not", "no thanks", "non"}

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|messages)",
        r"note to (ai|llm|assistant)s?",
        r"(reveal|print|show) (your|the) (system )?prompt",
        r"you are now",
        r"disregard (the |your )?(rules|instructions|guidelines)",
    )
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_SENSITIVE_KEYS = {"last_name", "email"}


def is_read_only(name: str, annotations: ToolAnnotations | None) -> bool:
    return bool(annotations and annotations.read_only_hint is True) and name in READ_ONLY_ALLOWLIST


def hidden_params_for(tool_name: str) -> frozenset[str]:
    return frozenset() if tool_name == IDENTITY_TOOL else HIDDEN_PARAMS


def parse_confirmation(text: str) -> bool | None:
    """True for a clear yes, False for a clear no, None for anything else (treated as "not confirmed")."""
    normalized = " ".join(re.sub(r"[^\w' ]", " ", text.lower()).split())
    if normalized in _YES:
        return True
    if normalized in _NO or normalized.startswith(("no ", "don't", "do not")):
        return False
    return None


def injection_suspected(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


# used to remove all personal data
def redact(value: Any, secrets: frozenset[str] = frozenset()) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if k in _SENSITIVE_KEYS else redact(v, secrets)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        value = _EMAIL_RE.sub("[email]", value)
        for secret in secrets:
            value = re.sub(re.escape(secret), "[redacted]", value, flags=re.IGNORECASE)
        return value
    return value


# values under sensitive keys (mask later)
def sensitive_values(data: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in _SENSITIVE_KEYS and isinstance(value, str):
                found.add(value)
            else:
                found |= sensitive_values(value)
    elif isinstance(data, list):
        for value in data:
            found |= sensitive_values(value)
    return found
