"""Hugging Face / Transformers Provider for Local Pipeline Execution.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
from banking_kyc.models import Address, ExtractedKYCData, KYCDocumentType, ProviderConfig
from banking_kyc.providers.base import BaseLLMProvider

class HuggingFaceProvider(BaseLLMProvider):
    """Inference client executing directly via Transformers and optional PEFT adapter."""

    _MODEL_CACHE: Dict[Tuple[str, str], Any] = {}

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._runtime = None

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        cache_key = (self.config.model_name, self.config.adapter_path or "")
        if cache_key in self._MODEL_CACHE:
            self._runtime = self._MODEL_CACHE[cache_key]
            return self._runtime
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            has_cuda = torch.cuda.is_available()
            if not has_cuda and self.config.adapter_path:
                self._runtime = False
                return self._runtime

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            load_kwargs: Dict[str, Any] = {
                "device_map": "auto" if has_cuda else "cpu",
                "torch_dtype": torch.bfloat16 if has_cuda else torch.float32,
            }
            if has_cuda and self.config.adapter_path:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **load_kwargs)
            if self.config.adapter_path and Path(self.config.adapter_path).is_dir():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.config.adapter_path, local_files_only=True)
            model.eval()
            self._runtime = (model, tokenizer)
            self._MODEL_CACHE[cache_key] = self._runtime
        except Exception:
            self._runtime = False
        return self._runtime

    def generate(self, prompt: str, **kwargs) -> str:
        runtime = self._get_runtime()
        if runtime:
            import torch
            model, tokenizer = runtime
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=False)
            return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

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
