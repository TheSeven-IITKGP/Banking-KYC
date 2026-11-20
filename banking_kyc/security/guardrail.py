"""Integrated Security Layer for Banking KYC.

Coordinates Inbound Prompt Guardrails and Outbound PII Redaction with an audit trail.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from banking_kyc.models import SecurityCheckResult
from banking_kyc.security.inbound_guard import PromptSecurityGuard
from banking_kyc.security.outbound_redactor import PIIRedactor


class SecurityLayer:
    """Production security controller managing inbound defense and outbound PII protection."""

    def __init__(self, enable_prompt_guard: bool = True, enable_pii_redaction: bool = True):
        self.enable_prompt_guard = enable_prompt_guard
        self.enable_pii_redaction = enable_pii_redaction
        self.inbound_guard = PromptSecurityGuard(strict_domain_check=True)
        self.redactor = PIIRedactor()
        self._audit_log: List[Dict[str, Any]] = []

    def inspect_inbound(self, text: str) -> SecurityCheckResult:
        """Inspects incoming document text or prompt."""
        if not self.enable_prompt_guard:
            return SecurityCheckResult(
                is_safe=True,
                risk_level="LOW",
                threat_type=None,
                details="Inbound guard disabled by admin configuration.",
                sanitized_input=text,
            )

        result = self.inbound_guard.analyze_input(text)
        if not result.is_safe:
            self._record_audit_event(
                event_type="SECURITY_THREAT_BLOCKED",
                details=result.details,
                threat_type=result.threat_type,
                risk_level=result.risk_level,
            )
        return result

    def sanitize_outbound_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes customer-facing or logging payload to strip cleartext PII."""
        if not self.enable_pii_redaction:
            return data
        return self.redactor.redact_dictionary(data)

    def sanitize_outbound_text(self, text: str) -> str:
        if not self.enable_pii_redaction:
            return text
        return self.redactor.redact_text(text)

    def _record_audit_event(self, event_type: str, details: str, threat_type: str | None, risk_level: str) -> None:
        self._audit_log.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "threat_type": threat_type,
            "risk_level": risk_level,
            "details": details,
        })

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._audit_log[-limit:]
