"""
Manasitra Privacy Engine

This module keeps the emotional-AI pipeline local-first by redacting common
personal identifiers before persistence or model context construction and by
returning auditable privacy metadata for every chat turn.
"""

import hashlib
import re
from dataclasses import dataclass


PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "upi_id": re.compile(r"\b[a-zA-Z0-9.\-_]{2,64}@(?!gmail|yahoo|outlook|hotmail|icloud|proton)[a-zA-Z]{2,64}\b", re.IGNORECASE),
    "pan_card": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)"),
    "aadhaar_like": re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
    "long_number": re.compile(r"(?<!\d)\d{7,}(?!\d)"),
}


@dataclass(frozen=True)
class PrivacyResult:
    redacted_text: str
    redaction_counts: dict[str, int]
    local_only: bool
    fingerprint: str


def privacy_filter(text: str) -> PrivacyResult:
    """Redact common PII and return an audit-safe fingerprint."""
    redacted = text
    counts: dict[str, int] = {}

    for label, pattern in PII_PATTERNS.items():
        redacted, count = pattern.subn(f"[REDACTED_{label.upper()}]", redacted)
        counts[label] = count

    fingerprint = hashlib.sha256(redacted.encode("utf-8")).hexdigest()[:16]
    return PrivacyResult(
        redacted_text=redacted,
        redaction_counts={k: v for k, v in counts.items() if v > 0},
        local_only=True,
        fingerprint=fingerprint,
    )


def privacy_metadata(result: PrivacyResult) -> dict:
    return {
        "local_only": result.local_only,
        "pii_redacted": bool(result.redaction_counts),
        "redaction_counts": result.redaction_counts,
        "message_fingerprint": result.fingerprint,
        "external_api_used": False,
    }

