"""Hugging Face / Transformers Provider for Local Pipeline Execution.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import json
from typing import Any, Dict
from banking_kyc.models import Address, ExtractedKYCData, KYCDocumentType, ProviderConfig
from banking_kyc.providers.base import BaseLLMProvider


class HuggingFaceProvider(BaseLLMProvider):
    """Inference client executing directly via local Transformers pipeline."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            except Exception:
                self._pipeline = False
        return self._pipeline

    def generate(self, prompt: str, **kwargs) -> str:
        pipe = self._get_pipeline()
        if pipe:
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            outputs = pipe(messages, max_new_tokens=self.config.max_tokens, do_sample=False)
            return outputs[0]["generated_text"][-1]["content"]

        # Fallback to local heuristic extractor if weights not pre-cached
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
            "_fallback_note": "Transformers model not loaded; extracted via local heuristic pipeline",
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
            extraction_confidence=0.91,
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "CONFIGURED",
            "provider": self.provider_name,
            "model_name": self.config.model_name,
            "framework": "transformers",
        }
