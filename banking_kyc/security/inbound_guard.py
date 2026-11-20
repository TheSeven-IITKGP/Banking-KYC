"""Inbound Prompt Guard & Domain Context Classifier.

Protects the KYC verification pipeline against prompt injections, jailbreaks,
system prompt leakage, and out-of-domain requests.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import re
from typing import List, Tuple
from banking_kyc.models import SecurityCheckResult


class PromptSecurityGuard:
    """Lightweight security classifier detecting prompt injection and validating KYC context."""

    # Adversarial & injection signatures
    INJECTION_PATTERNS = [
        (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE), "PROMPT_INJECTION_OVERRIDE"),
        (re.compile(r"system\s*prompt|system\s*instructions?|reveal\s+(the\s+)?instructions?", re.IGNORECASE), "SYSTEM_PROMPT_EXTRACTION"),
        (re.compile(r"bypass\s+(verification|kyc|security|validation|approval)", re.IGNORECASE), "VERIFICATION_BYPASS_ATTEMPT"),
        (re.compile(r"dan\s+mode|jailbreak|developer\s+mode|unrestricted\s+mode", re.IGNORECASE), "JAILBREAK_ATTEMPT"),
        (re.compile(r"<script.*?>|javascript:|onload=|onerror=", re.IGNORECASE), "XSS_ATTACK_VECTOR"),
        (re.compile(r"DROP\s+TABLE|SELECT\s+\*\s+FROM|UNION\s+SELECT|--|\bOR\b\s+1\s*=\s*1", re.IGNORECASE), "SQL_INJECTION"),
        (re.compile(r"you\s+are\s+now|act\s+as\s+a\s+hacker|disregard\s+rules", re.IGNORECASE), "ROLEPLAY_JAILBREAK"),
        (re.compile(r"always\s+return\s+(true|APPROVED|AUTO_APPROVED)", re.IGNORECASE), "FORCED_DECISION_INJECTION"),
    ]

    # Required banking/identity context anchors
    KYC_CONTEXT_KEYWORDS = [
        "AADHAAR", "PAN", "VOTER", "PASSPORT", "LICENCE", "LICENSE",
        "GOVERNMENT OF INDIA", "INCOME TAX", "UIDAI", "ELECTION",
        "DOB", "DATE OF BIRTH", "MALE", "FEMALE", "ADDRESS",
        "VERIFICATION", "KYC", "IDENTIFICATION", "DOCUMENT", "APPLICANT",
    ]

    def __init__(self, strict_domain_check: bool = True):
        self.strict_domain_check = strict_domain_check

    def analyze_input(self, text: str) -> SecurityCheckResult:
        """Evaluates input text for malicious payloads and KYC domain relevance."""
        if not text or not text.strip():
            return SecurityCheckResult(
                is_safe=False,
                risk_level="HIGH",
                threat_type="EMPTY_PAYLOAD",
                details="Empty document text or prompt provided.",
            )

        # 1. Check for prompt injection signatures
        for pattern, threat_type in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return SecurityCheckResult(
                    is_safe=False,
                    risk_level="CRITICAL",
                    threat_type=threat_type,
                    details=f"Detected adversarial attack or injection pattern: {threat_type}",
                )

        # 2. Check for out-of-domain content if strict check enabled
        if self.strict_domain_check:
            text_upper = text.upper()
            has_domain_anchor = any(kw in text_upper for kw in self.KYC_CONTEXT_KEYWORDS)
            # If text is long (> 50 chars) and completely lacks any KYC anchor
            if len(text) > 50 and not has_domain_anchor:
                return SecurityCheckResult(
                    is_safe=False,
                    risk_level="MEDIUM",
                    threat_type="OUT_OF_DOMAIN_PROMPT",
                    details="Input text lacks recognizable Indian KYC or identity document terminology.",
                )

        # 3. Passed inbound check
        return SecurityCheckResult(
            is_safe=True,
            risk_level="LOW",
            threat_type=None,
            details="Context valid; no adversarial or injection pattern detected.",
            sanitized_input=text,
        )
