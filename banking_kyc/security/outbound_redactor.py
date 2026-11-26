"""Outbound PII Redaction and Data Sanitization.

Ensures no raw customer PII (such as full unmasked Aadhaar or cleartext mobile numbers)
leaks into external logs, client responses, or test dashboards.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import re
from typing import Any, Dict

class PIIRedactor:
    """Masks Indian national identity numbers, phones, and sensitive identifiers."""

    AADHAAR_FULL_PATTERN = re.compile(r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b")
    AADHAAR_PARTIAL_PATTERN = re.compile(r"\bXXXX[- ]?XXXX[- ]?(\d{4})\b", re.IGNORECASE)

    PAN_PATTERN = re.compile(r"\b([A-Z]{3})([A-Z]{2}\d{2})(\d{2}[A-Z])\b")

    MOBILE_PATTERN = re.compile(r"(\+91[- ]?)?([6-9]\d{9})")

    EMAIL_PATTERN = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b")

    @classmethod
    def mask_aadhaar(cls, val: str) -> str:
        if not val:
            return val
        if cls.AADHAAR_PARTIAL_PATTERN.search(val):
            return cls.AADHAAR_PARTIAL_PATTERN.sub(r"XXXX-XXXX-\1", val)
        return cls.AADHAAR_FULL_PATTERN.sub(r"XXXX-XXXX-\3", val)

    @classmethod
    def mask_pan(cls, val: str) -> str:
        if not val:
            return val
        return cls.PAN_PATTERN.sub(r"\1XXXX\3", val)

    @classmethod
    def mask_mobile(cls, val: str) -> str:
        if not val:
            return val

        def repl(match: re.Match) -> str:
            prefix = match.group(1) or "+91-"
            digits = match.group(2)
            return f"{prefix}XXXXXX{digits[-4:]}"

        return cls.MOBILE_PATTERN.sub(repl, val)

    @classmethod
    def mask_email(cls, val: str) -> str:
        if not val:
            return val

        def repl(match: re.Match) -> str:
            local = match.group(1)
            domain = match.group(2)
            masked_local = local[0] + "***" if len(local) > 1 else "*"
            return f"{masked_local}@{domain}"

        return cls.EMAIL_PATTERN.sub(repl, val)

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Applies all PII masking rules to a plain text string."""
        if not text:
            return text
        t = cls.mask_aadhaar(text)
        t = cls.mask_mobile(t)
        t = cls.mask_email(t)
        return t

    @classmethod
    def redact_dictionary(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively traverses a dictionary and redacts sensitive PII fields."""
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                sanitized[k] = cls.redact_dictionary(v)
            elif isinstance(v, list):
                sanitized[k] = [cls.redact_dictionary(item) if isinstance(item, dict) else item for item in v]
            elif isinstance(v, str):
                k_lower = k.lower()
                if "aadhaar" in k_lower or ("document_number" in k_lower and re.search(r"\d{4}", v)):
                    sanitized[k] = cls.mask_aadhaar(v)
                elif "pan" in k_lower:
                    sanitized[k] = cls.mask_pan(v)
                elif "mobile" in k_lower or "phone" in k_lower:
                    sanitized[k] = cls.mask_mobile(v)
                elif "email" in k_lower:
                    sanitized[k] = cls.mask_email(v)
                else:
                    sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized
