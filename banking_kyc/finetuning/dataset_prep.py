"""Dataset Preparation and Instruction-Tuning Formatting for Indian KYC.

Splits the 2,000 synthetic Indian KYC dataset into 1,400 Train, 300 Val, 300 Test splits
and serializes them into Llama-3 (ChatML), Mistral, and Alpaca instruction formats.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple


class KYCDatasetPreparer:
    """Prepares instruction-tuning datasets from raw KYC JSONL records."""

    SYSTEM_PROMPT = (
        "You are an expert Indian Banking KYC Document Extraction Engine. "
        "Extract the identity details from the raw OCR text into valid JSON format with keys: "
        "document_type, document_number, full_name, father_name, date_of_birth, gender, and address."
    )

    def __init__(self, raw_jsonl_path: Path | str):
        self.raw_jsonl_path = Path(raw_jsonl_path)
        self.records: List[Dict[str, Any]] = []

    def load_records(self) -> List[Dict[str, Any]]:
        self.records = []
        with open(self.raw_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        return self.records

    def create_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.records:
            self.load_records()

        random.seed(seed)
        shuffled = list(self.records)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]

        return train_data, val_data, test_data

    @classmethod
    def format_llama3_chatml(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a single record into Llama-3 instruction chat format."""
        raw_ocr = record["ocr_input"]["raw_ocr_text"]
        gt_json = json.dumps(record["ground_truth_extraction_json"], ensure_ascii=False)

        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{cls.SYSTEM_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Extract KYC information from this document scan:\n\n{raw_ocr}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{gt_json}<|eot_id|>"
        )
        return {
            "record_id": record["record_id"],
            "document_type": record["document_type"],
            "scenario_tag": record.get("scenario_tag", "UNKNOWN"),
            "formatted_text": text,
        }

    @classmethod
    def format_alpaca(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a record into standard Alpaca format (instruction, input, output)."""
        return {
            "instruction": cls.SYSTEM_PROMPT,
            "input": record["ocr_input"]["raw_ocr_text"],
            "output": json.dumps(record["ground_truth_extraction_json"], ensure_ascii=False),
            "record_id": record["record_id"],
        }

    def export_splits(self, output_dir: Path | str) -> Dict[str, int]:
        """Processes and exports train/val/test splits to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train, val, test = self.create_splits()

        splits = {"train": train, "val": val, "test": test}
        summary = {}

        for split_name, data in splits.items():
            # Export raw JSONL
            jsonl_file = out_path / f"kyc_{split_name}.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            # Export Llama-3 ChatML format
            llama3_file = out_path / f"kyc_{split_name}_llama3.jsonl"
            with open(llama3_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(self.format_llama3_chatml(item)) + "\n")

            # Export Alpaca format
            alpaca_file = out_path / f"kyc_{split_name}_alpaca.jsonl"
            with open(alpaca_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(self.format_alpaca(item)) + "\n")

            summary[split_name] = len(data)

        return summary
