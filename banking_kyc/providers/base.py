"""Abstract Base Provider for LLM/VLM Inference.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Dict, Optional
from banking_kyc.models import ExtractedKYCData, KYCDocumentType, ProviderConfig

class BaseLLMProvider(abc.ABC):
    """Abstract interface all KYC extraction providers must implement."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def provider_name(self) -> str:
        return self.config.provider_type.value

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generates raw text response for a given prompt."""
        pass

    @abc.abstractmethod
    def extract_kyc_data(self, ocr_text: str) -> ExtractedKYCData:
        """Extracts structured KYC JSON fields from raw OCR text."""
        pass

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Checks whether the provider endpoint or weights are reachable."""
        pass

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Indian Banking KYC Document Extraction Engine. "
            "Given the raw OCR text of an Indian identity document (Aadhaar, PAN, Voter ID, "
            "Driving Licence, or Passport), extract the fields strictly in JSON format with keys: "
            "document_type, document_number, full_name, father_name, date_of_birth (YYYY-MM-DD), "
            "gender (MALE/FEMALE), and address (line1, city, district, state, pincode). "
            "Do not include conversational filler or markdown codeblocks outside the JSON."
        )

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Robust parser for LLM JSON outputs, stripping Markdown fences if present."""
        cleaned = response_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\})", cleaned)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return {}
