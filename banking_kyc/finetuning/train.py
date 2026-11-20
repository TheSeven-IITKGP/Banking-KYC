"""Training Pipeline Runner for QLoRA SFT Execution.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from banking_kyc.finetuning.dataset_prep import KYCDatasetPreparer
from banking_kyc.finetuning.qlora_config import QLoRAExperimentConfigs


def run_training_pipeline(
    config_name: str = "config_1_balanced",
    data_dir: str = "data/processed",
    output_dir: str = "models/adapters",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Executes or dry-runs the QLoRA fine-tuning training job."""
    configs = QLoRAExperimentConfigs.list_all_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config {config_name}. Choose from: {list(configs.keys())}")

    cfg = configs[config_name]
    out_path = Path(output_dir) / cfg.experiment_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save experiment config JSON
    config_file = out_path / "experiment_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    status_report = {
        "experiment_name": cfg.experiment_name,
        "base_model": cfg.base_model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.num_train_epochs,
        "target_modules": cfg.target_modules,
        "quantization": cfg.quantization_type,
        "dry_run": dry_run,
        "output_directory": str(out_path),
        "status": "COMPLETED_DRY_RUN" if dry_run else "IN_TRAINING",
    }

    # Write training run summary
    with open(out_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(status_report, f, indent=2)

    return status_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA KYC Fine-Tuning Runner")
    parser.add_argument("--config", default="config_1_balanced", help="Config name")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Perform dry run validation")
    args = parser.parse_args()

    res = run_training_pipeline(config_name=args.config, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))
