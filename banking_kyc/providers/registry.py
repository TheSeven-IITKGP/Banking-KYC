"""Provider Registry and Factory for Runtime Hot-Swapping.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

from typing import Dict, List, Optional
from banking_kyc.models import ProviderConfig, ProviderType
from banking_kyc.providers.base import BaseLLMProvider
from banking_kyc.providers.fine_tuned_local_provider import FineTunedLocalProvider
from banking_kyc.providers.hf_provider import HuggingFaceProvider
from banking_kyc.providers.ollama_provider import OllamaProvider
from banking_kyc.providers.vllm_provider import VLLMProvider


class ProviderRegistry:
    """Manages active and available LLM/VLM providers."""

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._active_type: ProviderType = ProviderType.LOCAL_FINE_TUNED
        self._initialize_defaults()

    def _initialize_defaults(self):
        # Register standard providers
        self.register(FineTunedLocalProvider())
        self.register(VLLMProvider(ProviderConfig(provider_type=ProviderType.VLLM, base_url="http://localhost:8000/v1")))
        self.register(OllamaProvider(ProviderConfig(provider_type=ProviderType.OLLAMA, base_url="http://localhost:11434")))
        self.register(HuggingFaceProvider(ProviderConfig(provider_type=ProviderType.HUGGINGFACE, model_name="Qwen/Qwen2.5-7B-Instruct")))

    def register(self, provider: BaseLLMProvider) -> None:
        self._providers[provider.provider_name] = provider

    def set_active_provider(self, provider_type: ProviderType | str) -> BaseLLMProvider:
        key = provider_type.value if isinstance(provider_type, ProviderType) else str(provider_type)
        if key not in self._providers:
            raise ValueError(f"Unknown provider: {key}. Available: {list(self._providers.keys())}")
        self._active_type = ProviderType(key)
        return self._providers[key]

    def get_active_provider(self) -> BaseLLMProvider:
        return self._providers[self._active_type.value]

    def list_providers(self) -> List[Dict[str, any]]:
        results = []
        for name, prov in self._providers.items():
            results.append({
                "name": name,
                "is_active": (name == self._active_type.value),
                "model_name": prov.config.model_name,
                "base_url": prov.config.base_url,
                "health": prov.health_check(),
            })
        return results

    def update_provider_config(self, provider_type: ProviderType | str, config_updates: dict) -> None:
        key = provider_type.value if isinstance(provider_type, ProviderType) else str(provider_type)
        if key in self._providers:
            current_cfg = self._providers[key].config.model_dump()
            current_cfg.update(config_updates)
            new_cfg = ProviderConfig(**current_cfg)
            if key == ProviderType.VLLM.value:
                self._providers[key] = VLLMProvider(new_cfg)
            elif key == ProviderType.OLLAMA.value:
                self._providers[key] = OllamaProvider(new_cfg)
            elif key == ProviderType.HUGGINGFACE.value:
                self._providers[key] = HuggingFaceProvider(new_cfg)
            elif key == ProviderType.LOCAL_FINE_TUNED.value:
                self._providers[key] = FineTunedLocalProvider(new_cfg)


# Global Singleton Registry
registry = ProviderRegistry()
