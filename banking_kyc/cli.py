"""Unified Command-Line Interface for Banking KYC System.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from banking_kyc.api.server import run_server
from banking_kyc.finetuning.dataset_prep import KYCDatasetPreparer
from banking_kyc.finetuning.evaluator import KYCEvaluator
from banking_kyc.finetuning.qlora_config import QLoRAExperimentConfigs
from banking_kyc.finetuning.train import run_training_pipeline
from banking_kyc.providers.registry import registry
from banking_kyc.workflow.graph import KYCWorkflowState, LangGraphKYCWorkflow


def cmd_serve(args):
    run_server(host=args.host, port=args.port)


def cmd_prepare_data(args):
    raw_path = Path(args.input)
    if not raw_path.exists():
        raw_path = Path(__file__).resolve().parent.parent / "synthetic_indian_kyc_ocr_finetune_2000.jsonl"
    print(f"📦 Loading KYC records from {raw_path}...")
    preparer = KYCDatasetPreparer(raw_path)
    summary = preparer.export_splits(args.output_dir)
    print("✅ Successfully generated instruction-tuning dataset splits:")
    for split, count in summary.items():
        print(f"   • {split}: {count} records (Llama-3 ChatML & Alpaca formats)")


def cmd_evaluate(args):
    print("📊 Evaluating Small-Model KYC Extraction Performance across benchmark matrix...")
    matrix = KYCEvaluator.get_benchmark_report()
    for model_name, metrics in matrix.items():
        print(f"\n[{model_name}]")
        print(f"  • Exact Match: {metrics['exact_match_pct']}%")
        print(f"  • Document Number Acc: {metrics['document_number_acc']}%")
        print(f"  • Name F1: {metrics['name_f1']}")
        print(f"  • Valid JSON: {metrics['valid_json_pct']}%")
        print(f"  • Hallucination Rate: {metrics['hallucination_rate_pct']}%")
        print(f"  • Latency: {metrics['mean_latency_ms']} ms")


def cmd_train(args):
    print(f"🚀 Initializing QLoRA fine-tuning training job for config: {args.config}")
    res = run_training_pipeline(config_name=args.config, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


def cmd_list_providers(args):
    print("🔌 Available Inference Providers:")
    for p in registry.list_providers():
        active_mark = " (ACTIVE)" if p["is_active"] else ""
        print(f"  • {p['name']}{active_mark}: Model={p['model_name']}, BaseURL={p['base_url']}")


def main():
    parser = argparse.ArgumentParser(description="FinKYC Production Platform CLI - Developed by Chandan Bauri")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Launch Production API & Web UI Server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host address")
    p_serve.add_argument("--port", type=int, default=8080, help="Port")
    p_serve.set_defaults(func=cmd_serve)

    # Prepare Data
    p_prep = subparsers.add_parser("prepare-data", help="Prepare train/val/test instruction splits")
    p_prep.add_argument("--input", default="synthetic_indian_kyc_ocr_finetune_2000.jsonl", help="Input JSONL path")
    p_prep.add_argument("--output-dir", default="data/processed", help="Output directory")
    p_prep.set_defaults(func=cmd_prepare_data)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation across models and ablations")
    p_eval.set_defaults(func=cmd_evaluate)

    # Train
    p_train = subparsers.add_parser("train", help="Run QLoRA fine-tuning pipeline")
    p_train.add_argument("--config", default="config_1_balanced", choices=list(QLoRAExperimentConfigs.list_all_configs().keys()))
    p_train.add_argument("--dry-run", action="store_true", default=True)
    p_train.set_defaults(func=cmd_train)

    # List Providers
    p_prov = subparsers.add_parser("list-providers", help="List configured providers")
    p_prov.set_defaults(func=cmd_list_providers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
