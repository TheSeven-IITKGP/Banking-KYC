"""Configuration settings for Banking KYC system.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field

from banking_kyc.models import ProviderConfig, ProviderType


class SystemThresholds(BaseModel):
    min_ocr_confidence: float = 0.70
    name_similarity_threshold: float = 0.80
    fuzzy_approval_threshold: float = 0.80
    strict_dob_match: bool = True
    require_exact_id_match: bool = True


class SecuritySettings(BaseModel):
    enable_prompt_guard: bool = True
    enable_pii_redaction: bool = True
    strict_domain_validation: bool = True
    mask_aadhaar_first_digits: int = 8  # XXXX-XXXX-1234
    mask_pan_middle_chars: bool = True


class KYCConfig(BaseModel):
    environment: str = "production"
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    active_provider: ProviderType = ProviderType.LOCAL_FINE_TUNED
    providers: Dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            ProviderType.LOCAL_FINE_TUNED.value: ProviderConfig(
                provider_type=ProviderType.LOCAL_FINE_TUNED,
                model_name="Llama-3-8B-KYC-QLoRA-r16",
                base_url=None,
            ),
            ProviderType.VLLM.value: ProviderConfig(
                provider_type=ProviderType.VLLM,
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                base_url="http://localhost:8000/v1",
            ),
            ProviderType.OLLAMA.value: ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                model_name="llama3:latest",
                base_url="http://localhost:11434",
            ),
            ProviderType.HUGGINGFACE.value: ProviderConfig(
                provider_type=ProviderType.HUGGINGFACE,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                base_url=None,
            ),
        }
    )
    thresholds: SystemThresholds = Field(default_factory=SystemThresholds)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    server_host: str = "0.0.0.0"
    server_port: int = 8080


# Global singleton configuration store
_global_config: KYCConfig = KYCConfig()


def get_config() -> KYCConfig:
    global _global_config
    return _global_config


def update_config(updated_dict: dict) -> KYCConfig:
    global _global_config
    current = _global_config.model_dump()
    for k, v in updated_dict.items():
        if k in current and isinstance(v, dict) and isinstance(current[k], dict):
            current[k].update(v)
        else:
            current[k] = v
    _global_config = KYCConfig(**current)
    return _global_config
