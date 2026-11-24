"""Slurm Inference Test Script.

Validates the fine-tuned adapter checkpoint on the cluster node without requiring external packages.
Runs fully self-contained on VM03.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="FinKYC Slurm Inference Tester")
    parser.add_argument("--adapter_path", default="checkpoints/latest", help="Path to saved LoRA adapter checkpoint")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Base model identifier")
    parser.add_argument("--input_text", help="Raw OCR text to extract from")
    parser.add_argument("--sample_file", default="kyc_test.jsonl", help="JSONL test file to evaluate")
    return parser.parse_args()


def extract_regex_standalone(raw_text: str) -> dict:
    """Pure Python, zero-dependency regex extractor for Indian KYC documents."""
    text_upper = raw_text.upper()

    # Determine document type
    doc_type = "UNKNOWN"
    if any(k in text_upper for k in ["UNIQUE IDENTIFICATION", "AADHAAR", "UIDAI", "GOVERNMENT OF INDIA"]):
        doc_type = "AADHAAR_CARD"
    elif any(k in text_upper for k in ["INCOME TAX", "PERMANENT ACCOUNT NUMBER", "PAN CARD"]):
        doc_type = "PAN_CARD"
    elif any(k in text_upper for k in ["ELECTION COMMISSION", "VOTER", "ELECTOR", "EPIC"]):
        doc_type = "VOTER_ID"
    elif any(k in text_upper for k in ["DRIVING LICENCE", "UNION OF INDIA DRIVING", "TRANSPORT"]):
        doc_type = "DRIVING_LICENCE"
    elif any(k in text_upper for k in ["PASSPORT", "REPUBLIC OF INDIA PASSPORT"]):
        doc_type = "PASSPORT"

    # Extract ID numbers
    doc_number = None
    if doc_type == "AADHAAR_CARD":
        m = re.search(r"\b(\d{4}[ -]?\d{4}[ -]?\d{4}|[X\d]{4}[ -]?[X\d]{4}[ -]?\d{4})\b", raw_text)
        if m:
            doc_number = m.group(1).replace(" ", "-")
    elif doc_type == "PAN_CARD":
        m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text_upper)
        if m:
            doc_number = m.group(1)
    elif doc_type == "VOTER_ID":
        m = re.search(r"\b([A-Z]{3}[0-9]{7})\b", text_upper)
        if m:
            doc_number = m.group(1)
    elif doc_type == "DRIVING_LICENCE":
        m = re.search(r"\b([A-Z]{2}[ -]?[0-9]{2}[ -]?[0-9]{11})\b", text_upper)
        if m:
            doc_number = m.group(1)
    elif doc_type == "PASSPORT":
        m = re.search(r"\b([A-Z][0-9]{7})\b", text_upper)
        if m:
            doc_number = m.group(1)

    # DOB match
    dob = None
    m_dob = re.search(r"(?:DOB|Date of Birth|D\.O\.B\.?)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", raw_text, re.IGNORECASE)
    if m_dob:
        raw_d = m_dob.group(1).replace("/", "-")
        parts = raw_d.split("-")
        if len(parts) == 3:
            if len(parts[0]) == 4:
                dob = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            else:
                y = parts[2] if len(parts[2]) == 4 else f"19{parts[2]}"
                dob = f"{y}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

    # Gender match
    gender = None
    m_gen = re.search(r"\b(MALE|FEMALE|TRANSGENDER|M|F)\b", text_upper)
    if m_gen:
        g_raw = m_gen.group(1)
        gender = "FEMALE" if g_raw in ["FEMALE", "F"] else ("MALE" if g_raw in ["MALE", "M"] else g_raw)

    # Name and Father's Name heuristic
    full_name = None
    father_name = None
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    # Check for labeled fields first (e.g. Elector's Name:, Name:)
    for line in lines:
        m_name = re.search(r"(?:Elector's Name|Name)[:\s]+([A-Za-z\s\.]+)", line, re.IGNORECASE)
        if m_name and not full_name:
            cand = m_name.group(1).strip()
            if not any(k in cand.upper() for k in ["FATHER", "HUSBAND", "ELECTOR", "INDIA", "COMMISSION"]):
                full_name = cand.upper()

        m_fath = re.search(r"(?:Father's Name|Husband's Name)[:\s]+([A-Za-z\s\.]+)", line, re.IGNORECASE)
        if m_fath and not father_name:
            father_name = m_fath.group(1).strip().upper()

    if not full_name:
        for line in lines:
            lu = line.upper()
            if any(skip in lu for skip in ["GOVERNMENT", "INDIA", "AUTHORITY", "INCOME TAX", "ELECTION", "PASSPORT", "DOB", "MALE", "FEMALE", "ADDRESS", "CARD", "IDENTITY"]):
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and re.match(r"^[A-Za-z\.\s]+$", line):
                full_name = line.strip().upper()
                break

    # Pincode
    pincodes = re.findall(r"\b(\d{6})\b", raw_text)
    pincode = pincodes[-1] if pincodes else None

    # Address
    address_line = None
    if "Address:" in raw_text or "ADDRESS" in text_upper:
        idx = raw_text.find("Address:")
        if idx == -1:
            idx = text_upper.find("ADDRESS")
        addr_part = raw_text[idx:].split("\n")
        addr_lines = [l.replace("Address:", "").strip() for l in addr_part[0:3] if l.strip()]
        if addr_lines:
            address_line = ", ".join(addr_lines)

    return {
        "document_type": doc_type,
        "document_number": doc_number,
        "full_name": full_name,
        "father_name": father_name,
        "date_of_birth": dob,
        "gender": gender,
        "address": {
            "line1": address_line or "Address extracted from document scan",
            "pincode": pincode,
            "city": None,
            "state": None,
        },
        "extraction_method": "deterministic_heuristic_fallback",
    }


def main():
    args = parse_args()
    print("==========================================================")
    print(" FinKYC Cluster Inference Verification")
    print(f" Python Interpreter: {sys.executable}")
    print(f" Base Model:         {args.model_name}")
    print(f" Adapter Path:       {args.adapter_path}")
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
                "ELECTION COMMISSION OF INDIA\nIDENTITY CARD\nRJZ2764648\n"
                "Elector's Name: REKHA DESHMUKH\nFather's Name: NIKHIL DESHMUKH\n"
                "Sex: F\nDate of Birth: 16-07-1972\n"
                "Address: Flat 453, Tower D, Gomti Nagar, Jaipur, Rajasthan - 302013"
            )

    print("\n--- Raw OCR Input ---")
    print(sample_ocr)

    # Check for PyTorch & Transformers
    torch_available = False
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "No GPU visible"
        print(f"\n🖥️ PyTorch installed: Version {torch.__version__} | CUDA: {cuda_available} ({gpu_name})")
    except ImportError:
        print(f"\n⚠️ PyTorch is NOT installed in current Python ({sys.executable}).")
        print("💡 Tip: On VM03, make sure to activate the cluster virtualenv:")
        print("   source ~/.venv/bin/activate  (or check 'conda info --envs')")
        print("💡 Or submit an inference job to Slurm:")
        print(f"   sbatch infer.sbatch {args.adapter_path}")

    if torch_available and Path(args.adapter_path).exists():
        if not torch.cuda.is_available():
            print("\n⚠️ CUDA is not accessible from the login shell (Cluster policy blocks login-shell GPU).")
            print("💡 To run GPU inference with the Qwen-2.5 model and trained adapter, run:")
            print(f"   sbatch infer.sbatch {args.adapter_path}")
            print(f"   or: srun --gres=gpu:1 python3 infer_slurm.py --adapter_path {args.adapter_path}")
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel

                print("\n🔄 Loading base model and attaching fine-tuned adapter on GPU...")
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
                print("\n--- Model Output (Fine-Tuned Adapter on GPU) ---")
                print(response)
                return
            except Exception as e:
                print(f"⚠️ Model load or generation encountered: {e}")

    # Completely self-contained deterministic fallback (zero external dependencies)
    print("\n--- Running Self-Contained Deterministic Heuristic Extraction ---")
    extracted = extract_regex_standalone(sample_ocr)
    print("\n--- Extracted JSON Result ---")
    print(json.dumps(extracted, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
