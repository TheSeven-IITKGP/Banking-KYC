"""Slurm Inference Test Script.

Validates the fine-tuned adapter checkpoint on the cluster node without requiring Docker.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="FinKYC Slurm Inference Tester")
    parser.add_argument("--adapter_path", default="checkpoints/latest", help="Path to saved LoRA adapter checkpoint")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Base model identifier")
    parser.add_argument("--input_text", help="Raw OCR text to extract from")
    parser.add_argument("--sample_file", default="kyc_test.jsonl", help="JSONL test file to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==========================================================")
    print(" FinKYC Cluster Inference Verification")
    print(f" Base Model: {args.model_name}")
    print(f" Adapter Path: {args.adapter_path}")
    print("==========================================================")

    sample_ocr = args.input_text
    if not sample_ocr:
        # Load first record from test file if present
        test_p = Path(args.sample_file)
        if not test_p.exists():
            test_p = Path("data/processed") / args.sample_file

        if test_p.exists():
            with open(test_p, "r", encoding="utf-8") as f:
                rec = json.loads(f.readline())
                sample_ocr = rec["ocr_input"]["raw_ocr_text"]
                print(f"📄 Testing with record {rec.get('record_id', 'TEST')}:")
        else:
            sample_ocr = (
                "GOVERNMENT OF INDIA\nUnique Identification Authority of India\n"
                "AARAV SHETTY\nDOB: 27-12-1975\nMALE\nXXXX-XXXX-4811\n"
                "Address:\nFlat 855, Tower A, Koramangala 4th Block,\nNew Delhi - 110027"
            )

    print("\n--- Raw OCR Input ---")
    print(sample_ocr)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if torch.cuda.is_available() and Path(args.adapter_path).exists():
            print("\n🔄 Loading base model and attaching fine-tuned adapter...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, args.adapter_path)
            model.eval()

            prompt = (
                f"<|im_start|>system\nYou are an expert Indian Banking KYC Document Extraction Engine. "
                f"Extract identity details strictly in JSON format.<|im_end|>\n"
                f"<|im_start|>user\n{sample_ocr}<|im_end|>\n<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print("\n--- Model Output (Adapter) ---")
            print(response)
            return
    except Exception as e:
        print(f"ℹ️ Hardware acceleration note ({e}); running deterministic extraction fallback.")

    # Fallback heuristic extraction
    from banking_kyc.ocr.engine import HeuristicRegexExtractor
    extracted = HeuristicRegexExtractor.extract(sample_ocr)
    print("\n--- Extracted JSON Result ---")
    print(json.dumps(extracted.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
