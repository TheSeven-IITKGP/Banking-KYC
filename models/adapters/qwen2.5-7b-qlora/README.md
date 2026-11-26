---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
tags:
- base_model:adapter:Qwen/Qwen2.5-7B-Instruct
- lora
- qlora
- transformers
- banking-kyc
- indian-kyc
- structured-extraction
---

# FinKYC Qwen2.5-7B-Instruct QLoRA Adapter (Job 178)

## Model Details

- **Developed by:** Chandan Bauri (<chandanbauri441@gmail.com>)
- **Model type:** Causal LM LoRA adapter (`PEFT 0.21.2`)
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Language:** English (Indian KYC identity scan text & JSON extraction)
- **License:** Apache 2.0 (subject to Qwen2.5 base model license terms)
- **Repository:** `Banking-KYC` (`models/adapters/qwen2.5-7b-qlora`)

## Uses

### Direct Use
Extracts structured JSON (`document_type`, `document_number`, `full_name`, `father_name`, `date_of_birth`, `gender`, `address`) from noisy OCR text of Indian identity documents (Aadhaar, PAN, Voter ID, Driving Licence, Passport).

### Downstream Use
Serves inside the `FinKYC` LangGraph workflow (`banking_kyc/workflow/graph.py`) alongside `PromptSecurityGuard`, `ImageQualityAnalyzer`, `PIIRedactor`, and the RBI Master Directions Hybrid RAG engine (`src/kycrag`).

### Out-of-Scope Use
Do not deploy for autonomous onboarding without deterministic ID/DOB verification, inbound prompt-injection filtering, and outbound PII masking. Do not run on blurry/glare-corrupted scans (`confidence < 0.70`) or non-Indian documents.

## Training Details

- **Training Data:** 1,400 stratified synthetic Indian KYC OCR training records (`data/processed/kyc_train.jsonl`), validated on 300 records and evaluated on 300 held-out test records.
- **Quantization:** 4-bit NormalFloat (`NF4`), double quantization, `bfloat16` compute dtype.
- **LoRA Hyperparameters:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- **Optimization:** 100 steps, learning rate `2e-4`, loss improved from `0.5906` to `0.1963` in `26.07s` on 1x NVIDIA L40S (`SLURM_JOB_ID=178`).
- **Artifact Size:** `adapter_model.safetensors` (`161,533,192` bytes).

## Evaluation Summary

| Configuration | Exact Match | Doc ID Acc | Hallucination Rate |
| :--- | :---: | :---: | :---: |
| Base Model | 54.2% | 71.0% | 14.8% |
| Base Model + RAG | 72.8% | 84.1% | 5.6% |
| Fine-Tuned Model (QLoRA r=16) | 98.4% | 99.7% | 0.3% |
| Fine-Tuned Model + RAG | 99.1% | 99.8% | 0.1% |

## Environmental Impact

- **Hardware Type:** 1x NVIDIA L40S (46 GB VRAM, Slurm cluster VM03)
- **Hours used:** 0.0072 GPU-hours (`26.07` seconds active training)

## How to Get Started

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quant, device_map="auto")
model = PeftModel.from_pretrained(base, "models/adapters/qwen2.5-7b-qlora")
tokenizer = AutoTokenizer.from_pretrained("models/adapters/qwen2.5-7b-qlora")
```

## Model Card Contact

- Chandan Bauri (<chandanbauri441@gmail.com>)
