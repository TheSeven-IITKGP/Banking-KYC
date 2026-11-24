"""Tests for Fine-Tuning Dataset Preparation and Evaluation Benchmarks.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from pathlib import Path
import unittest
from banking_kyc.finetuning.dataset_prep import KYCDatasetPreparer
from banking_kyc.finetuning.evaluator import KYCEvaluator
from banking_kyc.finetuning.qlora_config import QLoRAExperimentConfigs


class TestFineTuningAndEval(unittest.TestCase):

    def test_dataset_splits_and_formatting(self):
        jsonl_path = Path(__file__).resolve().parent.parent / "synthetic_indian_kyc_ocr_finetune_2000.jsonl"
        self.assertTrue(jsonl_path.exists())

        preparer = KYCDatasetPreparer(jsonl_path)
        records = preparer.load_records()
        self.assertEqual(len(records), 2000)

        train, val, test = preparer.create_splits(0.70, 0.15, 0.15)
        self.assertEqual(len(train), 1400)
        self.assertEqual(len(val), 300)
        self.assertEqual(len(test), 300)

        # Check Llama-3 ChatML format
        chatml = preparer.format_llama3_chatml(train[0])
        self.assertIn("<|begin_of_text|>", chatml["formatted_text"])
        self.assertIn("<|start_header_id|>system<|end_header_id|>", chatml["formatted_text"])

    def test_qlora_configurations(self):
        configs = QLoRAExperimentConfigs.list_all_configs()
        self.assertIn("config_1_balanced", configs)
        self.assertIn("config_2_expressive", configs)
        self.assertIn("ablation_1_low_rank", configs)
        self.assertIn("ablation_2_no_ocr_noise", configs)

        c1 = configs["config_1_balanced"]
        self.assertEqual(c1.lora_r, 16)
        self.assertEqual(c1.lora_alpha, 32)

    def test_benchmark_matrix(self):
        report = KYCEvaluator.get_benchmark_report()
        self.assertIn("Fine-Tuned Config 1 (QLoRA r=16, alpha=32)", report)
        self.assertGreater(report["Fine-Tuned Config 1 (QLoRA r=16, alpha=32)"]["exact_match_pct"], 95.0)

    def test_four_way_comparison(self):
        four_way = KYCEvaluator.get_four_way_comparison()
        self.assertIn("Base Model", four_way)
        self.assertIn("Base Model + RAG", four_way)
        self.assertIn("Fine-Tuned Model", four_way)
        self.assertIn("Fine-Tuned Model + RAG", four_way)
        self.assertGreater(four_way["Fine-Tuned Model + RAG"]["exact_match_pct"], 98.0)
        self.assertGreater(four_way["Fine-Tuned Model + RAG"]["judge_composite_score"], 9.0)

    def test_streamlit_app_and_finetuning_ui(self):
        ft_app_path = Path(__file__).resolve().parent.parent / "app" / "finetuning_streamlit_app.py"
        unified_app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
        self.assertTrue(ft_app_path.exists())
        self.assertTrue(unified_app_path.exists())

        # Verify FT_STATUS_LABEL matches the RAG color scheme
        content = unified_app_path.read_text(encoding="utf-8")
        self.assertIn("FT_STATUS_LABEL", content)
        self.assertIn("RAG_STATUS_LABEL", content)
        self.assertIn('"green"', content)
        self.assertIn('"orange"', content)
        self.assertIn('"red"', content)
        self.assertIn('"gray"', content)
        self.assertIn("KYC document verification", content)
        self.assertIn("KYC rules assistant", content)


if __name__ == "__main__":
    unittest.main()
