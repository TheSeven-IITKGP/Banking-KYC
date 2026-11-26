"""Ollama Provider for Local Small Open Model Execution.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict
from banking_kyc.models import Address, ExtractedKYCData, KYCDocumentType, ProviderConfig
from banking_kyc.providers.base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    """Inference client for Ollama local daemon."""

    def generate(self, prompt: str, **kwargs) -> str:
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        endpoint = f"{base_url}/api/chat"
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["message"]["content"]
        except Exception as e:
            from banking_kyc.ocr.engine import HeuristicRegexExtractor
            ext = HeuristicRegexExtractor.extract(prompt)
            return json.dumps({
                "document_type": ext.document_type.value,
                "document_number": ext.document_number,
                "full_name": ext.full_name,
                "father_name": ext.father_name,
                "date_of_birth": ext.date_of_birth,
                "gender": ext.gender,
                "address": ext.address.model_dump() if ext.address else None,
                "_fallback_note": f"Ollama daemon unreachable ({e}); extracted via local fallback",
            })

    def extract_kyc_data(self, ocr_text: str) -> ExtractedKYCData:
        resp = self.generate(ocr_text)
        data = self.parse_json_response(resp)
        if not data:
            from banking_kyc.ocr.engine import HeuristicRegexExtractor
            return HeuristicRegexExtractor.extract(ocr_text)

        doc_type_str = data.get("document_type", "UNKNOWN")
        try:
            doc_type = KYCDocumentType(doc_type_str)
        except ValueError:
            doc_type = KYCDocumentType.UNKNOWN

        addr = None
        if isinstance(data.get("address"), dict):
            addr = Address(**data["address"])

        return ExtractedKYCData(
            document_type=doc_type,
            document_number=data.get("document_number"),
            full_name=data.get("full_name"),
            father_name=data.get("father_name"),
            date_of_birth=data.get("date_of_birth"),
            gender=data.get("gender"),
            address=addr,
            raw_text=ocr_text,
            extraction_confidence=0.90,
        )

    def health_check(self) -> Dict[str, Any]:
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        endpoint = f"{base_url}/api/tags"
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"status": "HEALTHY", "provider": self.provider_name, "models": data.get("models", [])}
        except Exception as e:
            return {"status": "UNREACHABLE", "provider": self.provider_name, "error": str(e), "hint": "Ensure 'ollama serve' is running"}
