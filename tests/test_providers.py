"""Tests for Pluggable Multi-Provider Setup and Hot-Swapping.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

import unittest
from banking_kyc.models import ProviderType
from banking_kyc.providers.fine_tuned_local_provider import FineTunedLocalProvider
from banking_kyc.providers.registry import ProviderRegistry


class TestProviderRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ProviderRegistry()

    def test_default_active_provider(self):
        active = self.registry.get_active_provider()
        self.assertEqual(active.provider_name, ProviderType.LOCAL_FINE_TUNED.value)

    def test_switch_provider(self):
        self.registry.set_active_provider(ProviderType.VLLM)
        self.assertEqual(self.registry.get_active_provider().provider_name, ProviderType.VLLM.value)

        # Switch back to local
        self.registry.set_active_provider(ProviderType.LOCAL_FINE_TUNED)
        self.assertEqual(self.registry.get_active_provider().provider_name, ProviderType.LOCAL_FINE_TUNED.value)

    def test_local_provider_extraction(self):
        prov = FineTunedLocalProvider()
        health = prov.health_check()
        self.assertEqual(health["status"], "HEALTHY")

        text = "GOVERNMENT OF INDIA\nROHIT SHARMA\nDOB: 12-05-1987\nXXXX-XXXX-9999\nVID : 1234-5678-9012"
        extracted = prov.extract_kyc_data(text)
        self.assertEqual(extracted.document_number, "XXXX-XXXX-9999")
        self.assertEqual(extracted.full_name, "ROHIT SHARMA")


if __name__ == "__main__":
    unittest.main()
