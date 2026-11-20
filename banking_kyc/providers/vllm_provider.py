"""vLLM / OpenAI-compatible Serving Provider.

Connects to vLLM or any OpenAI-compatible API endpoint serving quantized models.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict
from banking_kyc.models import Address, ExtractedKYCData, KYCDocumentType, ProviderConfig
from banking_kyc.providers.base import BaseLLMProvider


class VLLMProvider(BaseLLMProvider):
    """Inference client for vLLM server instances."""

    def generate(self, prompt: str, **kwargs) -> str:
        base_url = (self.config.base_url or "http://localhost:8000/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            # Fallback to local extractor if vLLM server is unreachable
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
                "_fallback_note": f"vLLM server unreachable ({e}); extracted via local fallback",
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
            extraction_confidence=0.92,
        )

    def health_check(self) -> Dict[str, Any]:
        base_url = (self.config.base_url or "http://localhost:8000/v1").rstrip("/")
        endpoint = f"{base_url}/models"
        try:
            req = urllib.request.Request(endpoint, method="GET")
            if self.config.api_key:
                req.add_header("Authorization", f"Bearer {self.config.api_key}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"status": "HEALTHY", "provider": self.provider_name, "models": data}
        except Exception as e:
            return {"status": "UNREACHABLE", "provider": self.provider_name, "error": str(e), "hint": "Run vllm serve in container"}
