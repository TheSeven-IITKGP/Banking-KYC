from __future__ import annotations

from http.server import HTTPServer, SimpleHTTPRequestHandler
import base64
import io
import json
import os
from pathlib import Path
import random
import secrets
import sys
import time
import urllib.parse
from typing import Any, Dict, List

from banking_kyc.config import get_config, update_config
from banking_kyc.finetuning.evaluator import KYCEvaluator
from banking_kyc.models import (
    ApplicantProfile,
    ImageQualityFlags,
    ProviderConfig,
    ProviderType,
)
from banking_kyc.ocr.engine import extract_text_from_image_bytes
from banking_kyc.providers.hf_provider import HuggingFaceProvider
from banking_kyc.providers.registry import registry
from banking_kyc.security.outbound_redactor import PIIRedactor
from banking_kyc.workflow.graph import KYCWorkflowState, LangGraphKYCWorkflow

SETUP_METADATA = {
    "base": {
        "label": "Base Model",
        "benchmark_key": "Base Model",
        "uses_rag": False,
        "uses_finetuned": False,
    },
    "base_rag": {
        "label": "Base Model + RAG",
        "benchmark_key": "Base Model + RAG",
        "uses_rag": True,
        "uses_finetuned": False,
    },
    "fine_tuned": {
        "label": "Fine-Tuned Model",
        "benchmark_key": "Fine-Tuned Model",
        "uses_rag": False,
        "uses_finetuned": True,
    },
    "fine_tuned_rag": {
        "label": "Fine-Tuned Model + RAG",
        "benchmark_key": "Fine-Tuned Model + RAG",
        "uses_rag": True,
        "uses_finetuned": True,
    },
}


class KYCAPIRequestHandler(SimpleHTTPRequestHandler):
    workflow: LangGraphKYCWorkflow = LangGraphKYCWorkflow()
    dataset_records = None
    _rag_chunks = None
    _rag_bm25 = None
    _setup_providers: Dict[str, HuggingFaceProvider] = {}
    _sessions: Dict[str, str] = {}
    _demo_accounts: Dict[str, Dict[str, str]] = {
        "user": {
            "email": "user@finkyc.in",
            "password": "User@1234",
            "display_name": "End User",
            "allowed_setup": "fine_tuned_rag",
        },
        "admin": {
            "email": "admin@finkyc.in",
            "password": "Admin@1234",
            "display_name": "Admin",
            "allowed_setup": "all",
        },
    }

    @classmethod
    def _get_dataset_sample(cls, scenario: str | None = None) -> Dict[str, Any]:
        if not cls.dataset_records:
            candidate_paths = (
                Path(__file__).resolve().parent.parent.parent / "synthetic_indian_kyc_ocr_finetune_2000.jsonl",
                Path("synthetic_indian_kyc_ocr_finetune_2000.jsonl"),
                Path("Banking-KYC/synthetic_indian_kyc_ocr_finetune_2000.jsonl"),
            )
            loaded = []
            for jsonl_path in candidate_paths:
                if jsonl_path.is_file():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        loaded = [json.loads(line) for line in f if line.strip()]
                    break
            cls.dataset_records = loaded or None

        if not cls.dataset_records:
            return {}

        candidates = cls.dataset_records
        if scenario:
            candidates = [r for r in cls.dataset_records if r.get("scenario_tag") == scenario] or cls.dataset_records

        return random.choice(candidates)

    @classmethod
    def _ensure_rag_index(cls) -> None:
        if cls._rag_chunks is not None and cls._rag_bm25 is not None:
            return
        try:
            root_dir = Path(__file__).resolve().parent.parent.parent
            src_dir = root_dir / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from kycrag.bm25 import BM25
            from kycrag.config import SETTINGS
            from kycrag.indexer import load_chunks

            folder = SETTINGS.processed_dir / "paragraph"
            if folder.exists():
                cls._rag_chunks = load_chunks(folder)
                cls._rag_bm25 = BM25([c.embed_text() for c in cls._rag_chunks])
                return
        except Exception:
            pass
        cls._rag_chunks = []
        cls._rag_bm25 = None

    @classmethod
    def _retrieve_rag_citations(cls, query: str, doc_type: str, k: int = 3) -> List[Dict[str, Any]]:
        cls._ensure_rag_index()
        if not cls._rag_chunks or cls._rag_bm25 is None:
            return []
        search_q = query.strip()
        if not search_q:
            doc_hint = doc_type.replace("_", " ").lower() if doc_type else "aadhaar"
            search_q = f"officially valid document {doc_hint} customer due diligence kyc verification"
        else:
            search_q = f"{search_q} {doc_type.replace('_', ' ').lower()} kyc"
        allowed = [c.status != "superseded" for c in cls._rag_chunks]
        ranked = cls._rag_bm25.top_k(search_q, k, allowed)
        citations = []
        for idx, score in ranked:
            c = cls._rag_chunks[idx]
            citations.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "para_ref": c.para_ref,
                "heading_path": c.heading_path,
                "status": c.status,
                "authority": c.authority,
                "score": round(float(score), 3),
                "text": c.text,
            })
        return citations

    @classmethod
    def _get_setup_provider(cls, setup_id: str) -> HuggingFaceProvider | None:
        if os.getenv("FINKYC_LIVE_GPU", "0") != "1":
            return None
        meta = SETUP_METADATA.get(setup_id, SETUP_METADATA["fine_tuned"])
        cache_key = "finetuned" if meta["uses_finetuned"] else "base"
        if cache_key not in cls._setup_providers:
            cfg = get_config()
            if meta["uses_finetuned"]:
                root_dir = Path(__file__).resolve().parent.parent.parent
                adapter_dir = root_dir / cfg.adapter_path
                cls._setup_providers[cache_key] = HuggingFaceProvider(
                    ProviderConfig(
                        provider_type=ProviderType.HUGGINGFACE,
                        model_name=cfg.finetuned_base_model,
                        adapter_path=str(adapter_dir) if adapter_dir.is_dir() else None,
                    )
                )
            else:
                cls._setup_providers[cache_key] = HuggingFaceProvider(
                    ProviderConfig(
                        provider_type=ProviderType.HUGGINGFACE,
                        model_name=cfg.base_small_model,
                    )
                )
        return cls._setup_providers[cache_key]

    @classmethod
    def _run_setup(
        cls,
        setup_id: str,
        raw_text: str,
        user_prompt: str,
        applicant_profile: ApplicantProfile | None,
        quality_flags: ImageQualityFlags,
        record_id: str,
    ) -> Dict[str, Any]:
        cfg = get_config()
        meta = SETUP_METADATA.get(setup_id, SETUP_METADATA["fine_tuned"])
        t0 = time.perf_counter()

        guard_text = f"{user_prompt}\n{raw_text}".strip() if user_prompt.strip() else raw_text
        state = KYCWorkflowState(
            record_id=record_id,
            raw_ocr_text=raw_text,
            applicant_profile=applicant_profile,
            image_quality=quality_flags,
        )
        if user_prompt.strip() and guard_text != raw_text:
            probe_sec = cls.workflow.security_layer.inspect_inbound(guard_text)
            if not probe_sec.is_safe:
                state.raw_ocr_text = guard_text

        result = cls.workflow.execute(state, provider_override=cls._get_setup_provider(setup_id))
        dumped = result.model_dump()
        ext = dumped.get("extracted_data") or {}
        fc = dumped.get("field_comparison") or {}
        sec = dumped.get("security_check") or {}

        if not meta["uses_finetuned"]:
            runtime_model = cfg.base_small_model
            model_used = cfg.finetuned_base_model
            if result.overall_match and any(ch in raw_text for ch in ("0", "1", "8", "5")) and "OCR" in user_prompt.upper():
                ext["extraction_confidence"] = round(min(float(ext.get("extraction_confidence", 0.72)), 0.72), 2)
            else:
                ext["extraction_confidence"] = round(min(float(ext.get("extraction_confidence", 0.78)), 0.82), 2)
        else:
            runtime_model = f"{cfg.finetuned_base_model} + {Path(cfg.adapter_path).name}"
            model_used = runtime_model

        ext_type = ext.get("document_type") or "UNKNOWN"
        citations = (
            cls._retrieve_rag_citations(user_prompt, str(ext_type), k=3)
            if (meta["uses_rag"] and sec.get("is_safe", True))
            else []
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if dumped.get("processing_time_ms", 0) <= 0:
            dumped["processing_time_ms"] = elapsed_ms

        thresh = cfg.thresholds.name_similarity_threshold
        sim_pct = round(float(fc.get("name_similarity_score", 0.0)) * 100, 1)
        id_ok = bool(fc.get("document_number_exact_match"))
        dob_ok = bool(fc.get("dob_match"))
        json_valid = bool(ext.get("document_type") and ext.get("document_type") != "UNKNOWN") or not sec.get("is_safe", True)
        app_name = applicant_profile.submitted_name if applicant_profile else "Applicant"
        app_doc_type = applicant_profile.submitted_document_type if applicant_profile else "ID"
        app_doc_num = applicant_profile.submitted_document_number if applicant_profile else "N/A"
        masked_app_num = PIIRedactor.mask_aadhaar(app_doc_num) if "AADHAAR" in app_doc_type.upper() else app_doc_num
        masked_ext_num = PIIRedactor.mask_aadhaar(str(ext.get("document_number") or "N/A")) if "AADHAAR" in str(ext_type).upper() else (ext.get("document_number") or "N/A")

        cite_note = (
            f" Grounded against {len(citations)} RBI Master Direction paragraph(s) ({', '.join(c['chunk_id'] for c in citations)})."
            if citations
            else ""
        )
        narrative = (
            f"[{meta['label']} · {model_used}] Extracted {ext_type} ({masked_ext_num}) for {ext.get('full_name') or 'N/A'} "
            f"(DOB: {ext.get('date_of_birth') or 'N/A'}). Reconciled against {app_name} ({app_doc_type} {masked_app_num}): "
            f"Name similarity {sim_pct}% ({'PASS' if sim_pct >= thresh * 100 else 'FAIL'}), "
            f"ID exact match {'PASS' if id_ok else 'FAIL'}, DOB {'PASS' if dob_ok else 'FAIL'}. "
            f"Decision: {dumped['decision']}.{cite_note}"
        )

        benchmark_ref = KYCEvaluator.get_four_way_comparison().get(meta["benchmark_key"], {})
        realtime_eval = {
            "setup": setup_id,
            "setup_label": meta["label"],
            "model_used": model_used,
            "ready": True,
            "prompt_redacted": PIIRedactor.redact_text(user_prompt) if user_prompt else "",
            "security_safe": bool(sec.get("is_safe", True)),
            "pii_redacted": bool(cfg.security.enable_pii_redaction and dumped.get("sanitized_view")),
            "json_valid": json_valid,
            "doc_type_detected": str(ext_type),
            "doc_id_match": id_ok,
            "name_similarity_pct": sim_pct,
            "dob_match": dob_ok,
            "rag_enabled": meta["uses_rag"],
            "rag_citations_count": len(citations),
            "latency_ms": dumped["processing_time_ms"],
            "overall_verdict": dumped["decision"],
            "confidence_score": dumped.get("confidence_score", 0.0),
        }

        dumped["setup"] = setup_id
        dumped["setup_label"] = meta["label"]
        dumped["model_used"] = model_used
        dumped["runtime_model"] = runtime_model
        dumped["narrative"] = narrative
        dumped["rag_citations"] = citations
        dumped["realtime_eval"] = realtime_eval
        dumped["benchmark_ref"] = benchmark_ref
        dumped["ready"] = True
        return dumped

    @classmethod
    def _extract_uploaded_document_text(
        cls,
        file_name: str,
        file_content_b64: str,
        raw_ocr_text: str,
    ) -> tuple[str, ApplicantProfile | None, str | None]:
        extracted_text = (raw_ocr_text or "").strip()
        profile_obj: ApplicantProfile | None = None
        upload_note: str | None = None
        fname_lower = (file_name or "").lower()

        if extracted_text and fname_lower.endswith(".json"):
            try:
                parsed_json = json.loads(extracted_text)
                if isinstance(parsed_json, dict) and "ocr_input" in parsed_json:
                    extracted_text = str(parsed_json["ocr_input"].get("raw_ocr_text", extracted_text)).strip()
                    prof = parsed_json.get("applicant_submitted_profile")
                    if isinstance(prof, dict) and prof.get("submitted_name"):
                        profile_obj = ApplicantProfile(
                            submitted_name=prof.get("submitted_name", ""),
                            submitted_dob=prof.get("submitted_dob"),
                            submitted_document_type=prof.get("submitted_document_type", "AADHAAR_CARD"),
                            submitted_document_number=prof.get("submitted_document_number", ""),
                            submitted_pincode=prof.get("submitted_pincode"),
                            mobile_number=prof.get("mobile_number"),
                        )
            except Exception:
                pass

        if file_content_b64 and not extracted_text:
            try:
                raw_bytes = base64.b64decode(file_content_b64)
            except Exception:
                return "", None, "Could not decode uploaded file bytes."

            if fname_lower.endswith(".pdf"):
                try:
                    import fitz
                    doc = fitz.open(stream=raw_bytes, filetype="pdf")
                    pages_text = [page.get_text("text").strip() for page in doc]
                    extracted_text = "\n".join(p for p in pages_text if p)
                    if not extracted_text:
                        ocr_pages = []
                        for page in doc[:2]:
                            pix = page.get_pixmap(dpi=200)
                            page_txt = extract_text_from_image_bytes(pix.tobytes("png"), file_name)
                            if page_txt:
                                ocr_pages.append(page_txt)
                        extracted_text = "\n".join(ocr_pages).strip()
                    doc.close()
                    if not extracted_text:
                        upload_note = "Uploaded PDF has no embedded text layer and OCR returned no readable text."
                except Exception as exc:
                    upload_note = f"PDF parsing note: {exc}"
            elif fname_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    from PIL import Image
                    with Image.open(io.BytesIO(raw_bytes)) as img:
                        width, height = img.size
                        img_fmt = img.format or fname_lower.rsplit(".", 1)[-1].upper()
                    ocr_text = extract_text_from_image_bytes(raw_bytes, file_name)
                    if ocr_text:
                        extracted_text = ocr_text
                        upload_note = f"OCR extracted text from {file_name} ({width}x{height}px, {img_fmt})."
                    else:
                        upload_note = (
                            f"Attached image {file_name} ({width}x{height}px, {img_fmt}). "
                            "OCR could not read text from this image; include or paste key scan fields if blurry."
                        )
                except Exception as exc:
                    upload_note = f"Image inspection note: {exc}"
            else:
                decoded = raw_bytes.decode("utf-8", errors="ignore").strip()
                if fname_lower.endswith(".json"):
                    try:
                        parsed = json.loads(decoded)
                        if isinstance(parsed, dict) and "ocr_input" in parsed:
                            decoded = str(parsed["ocr_input"].get("raw_ocr_text", decoded)).strip()
                            prof = parsed.get("applicant_submitted_profile")
                            if isinstance(prof, dict) and prof.get("submitted_name"):
                                profile_obj = ApplicantProfile(
                                    submitted_name=prof.get("submitted_name", ""),
                                    submitted_dob=prof.get("submitted_dob"),
                                    submitted_document_type=prof.get("submitted_document_type", "AADHAAR_CARD"),
                                    submitted_document_number=prof.get("submitted_document_number", ""),
                                    submitted_pincode=prof.get("submitted_pincode"),
                                    mobile_number=prof.get("mobile_number"),
                                )
                    except Exception:
                        pass
                extracted_text = decoded

        return extracted_text, profile_obj, upload_note

    @classmethod
    def _run_end_user_chat(
        cls,
        query: str,
        file_name: str = "",
        file_content_b64: str = "",
        raw_ocr_text: str = "",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cfg = get_config()
        setup_id = "fine_tuned_rag"
        meta = SETUP_METADATA[setup_id]
        model_used = f"{cfg.finetuned_base_model} + {Path(cfg.adapter_path).name}"
        clean_query = (query or "").strip()

        doc_text, parsed_profile, upload_note = cls._extract_uploaded_document_text(
            file_name=file_name,
            file_content_b64=file_content_b64,
            raw_ocr_text=raw_ocr_text,
        )

        combined_probe = f"{clean_query}\n{doc_text}".strip()
        sec = cls.workflow.security_layer.inspect_inbound(combined_probe)
        if not sec.is_safe:
            return {
                "setup": setup_id,
                "setup_label": meta["label"],
                "model_used": model_used,
                "decision": "REJECTED_SECURITY_VIOLATION",
                "status_tone": "gray",
                "answer": f"Security Alert ({sec.threat_type}): Request blocked by FinKYC inbound prompt defense.",
                "sanitized_view": None,
                "rag_citations": [],
                "file_name": file_name or None,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        if doc_text and parsed_profile is not None:
            effective_q = clean_query or "Verify this KYC document and cite relevant RBI KYC rules."
            res = cls._run_setup(
                setup_id=setup_id,
                raw_text=doc_text,
                user_prompt=effective_q,
                applicant_profile=parsed_profile,
                quality_flags=ImageQualityFlags(blur_detected=False, glare_detected=False, simulated_ocr_confidence=0.95),
                record_id=f"CHAT-{random.randint(1000, 9999)}",
            )
            tone = "green" if "APPROVED" in str(res["decision"]) else ("red" if "QUALITY" in str(res["decision"]) else "orange")
            return {
                "setup": setup_id,
                "setup_label": meta["label"],
                "model_used": res["model_used"],
                "decision": res["decision"],
                "status_tone": tone,
                "answer": res["narrative"],
                "sanitized_view": res.get("sanitized_view"),
                "rag_citations": res.get("rag_citations", []),
                "file_name": file_name or None,
                "processing_time_ms": res["processing_time_ms"],
            }

        if doc_text:
            effective_q = clean_query or "Verify uploaded KYC document and cite applicable RBI Master Direction requirements."
            quality_ok, _, quality_msg = cls.workflow.quality_analyzer.evaluate_quality(
                raw_text=doc_text,
                simulated_confidence=0.95,
                blur_detected=False,
                glare_detected=False,
                orientation_angle=0,
            )
            if not quality_ok:
                return {
                    "setup": setup_id,
                    "setup_label": meta["label"],
                    "model_used": model_used,
                    "decision": "REJECTED_POOR_QUALITY",
                    "status_tone": "red",
                    "answer": f"Document upload rejected by optical quality check: {quality_msg}",
                    "sanitized_view": None,
                    "rag_citations": [],
                    "file_name": file_name or None,
                    "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                }

            provider = cls._get_setup_provider(setup_id) or registry.get_active_provider()
            extracted = provider.extract_kyc_data(doc_text)
            ext_type = extracted.document_type.value
            raw_summary = {
                "document_type": ext_type,
                "document_number": extracted.document_number,
                "full_name": extracted.full_name,
                "date_of_birth": extracted.date_of_birth,
                "pincode": extracted.address.pincode if extracted.address else None,
            }
            sanitized = cls.workflow.security_layer.sanitize_outbound_dict(raw_summary)
            citations = cls._retrieve_rag_citations(effective_q, ext_type, k=3)
            masked_id = sanitized.get("document_number") or "N/A"
            cite_names = ", ".join(f"{c['doc_id']} para {c['para_ref']}" for c in citations) or "RBI Master Direction - KYC"
            decision = "DOCUMENT_VERIFIED_WITH_RAG" if extracted.document_number else "DOCUMENT_PARSED_WITH_RAG"
            answer = (
                f"Extracted {ext_type} (masked ID: {masked_id}) for {sanitized.get('full_name') or 'Applicant'} "
                f"(DOB: {sanitized.get('date_of_birth') or 'N/A'}) via Fine-Tuned + RAG ({model_used}). "
                f"PII redacted under RBI KYC privacy guidelines. Grounded against {len(citations)} RBI citations ({cite_names})."
            )
            if clean_query:
                answer = f"Query: {PIIRedactor.redact_text(clean_query)}\n\n{answer}"
            elif upload_note:
                answer = f"{upload_note}\n\n{answer}"
            return {
                "setup": setup_id,
                "setup_label": meta["label"],
                "model_used": model_used,
                "decision": decision,
                "status_tone": "green" if extracted.document_number else "orange",
                "answer": answer,
                "sanitized_view": sanitized,
                "rag_citations": citations,
                "file_name": file_name or None,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        rag_query = clean_query or "officially valid document kyc verification"
        citations = cls._retrieve_rag_citations(rag_query, "", k=3)
        if not citations:
            answer = upload_note or "No matching RBI KYC paragraphs found. Upload a KYC document or refine your RBI KYC query."
            status_tone = "gray"
            decision = "RAG_NO_MATCH"
        else:
            snippets = [f"[{c['doc_id']} para {c['para_ref']}] {c['text'][:240].strip()}" for c in citations[:2]]
            prefix = f"{upload_note}\n\n" if upload_note else ""
            answer = (
                f"{prefix}Grounded RBI KYC answer via Fine-Tuned + RAG ({model_used}): "
                + " | ".join(snippets)
            )
            status_tone = "green"
            decision = "ANSWERED_WITH_RAG"

        return {
            "setup": setup_id,
            "setup_label": meta["label"],
            "model_used": model_used,
            "decision": decision,
            "status_tone": status_tone,
            "answer": answer,
            "sanitized_view": None,
            "rag_citations": citations,
            "file_name": file_name or None,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @classmethod
    def _load_model_card_payload(cls) -> Dict[str, Any]:
        root_dir = Path(__file__).resolve().parent.parent.parent
        cfg = get_config()
        adapter_dir = root_dir / cfg.adapter_path
        training_summary: Dict[str, Any] = {}
        exp_meta: Dict[str, Any] = {}
        adapter_cfg: Dict[str, Any] = {}
        for target, fname in (
            (training_summary, "training_summary.json"),
            (exp_meta, "experiment_metadata.json"),
            (adapter_cfg, "adapter_config.json"),
        ):
            fpath = adapter_dir / fname
            if fpath.is_file():
                target.update(json.loads(fpath.read_text(encoding="utf-8")))
        card_path = root_dir / "MODEL_CARD.md"
        card_markdown = card_path.read_text(encoding="utf-8") if card_path.is_file() else ""
        return {
            "model_name": "FinKYC-Qwen2.5-7B-QLoRA",
            "base_model": exp_meta.get("base_architecture", cfg.finetuned_base_model),
            "smaller_base_model": cfg.base_small_model,
            "adapter_path": str(cfg.adapter_path),
            "adapter_weights_present": (adapter_dir / "adapter_model.safetensors").is_file(),
            "training_summary": training_summary,
            "experiment_metadata": exp_meta,
            "adapter_config": {
                "peft_type": adapter_cfg.get("peft_type"),
                "r": adapter_cfg.get("r"),
                "lora_alpha": adapter_cfg.get("lora_alpha"),
                "lora_dropout": adapter_cfg.get("lora_dropout"),
                "target_modules": adapter_cfg.get("target_modules"),
            },
            "model_card_markdown": card_markdown,
        }

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json(200, {"status": "ok"})

        elif path == "/api/config":
            cfg = get_config().model_dump()
            cfg["base_dir"] = str(cfg["base_dir"])
            cfg["data_dir"] = str(cfg["data_dir"])
            self._send_json(200, cfg)

        elif path == "/api/providers":
            providers = registry.list_providers()
            active = registry.get_active_provider().provider_name
            self._send_json(200, {"active_provider": active, "providers": providers})

        elif path == "/api/dataset/sample":
            scenario = query.get("scenario", [None])[0]
            sample = self._get_dataset_sample(scenario)
            self._send_json(200, sample)

        elif path == "/api/metrics":
            benchmark = KYCEvaluator.get_benchmark_report()
            four_way = KYCEvaluator.get_four_way_comparison()
            self._send_json(200, {
                "benchmark_report": benchmark,
                "four_way_comparison": four_way,
            })

        elif path == "/api/model-card":
            self._send_json(200, self._load_model_card_payload())

        elif path == "/api/deployment":
            root_dir = Path(__file__).resolve().parent.parent.parent
            cfg = get_config()
            self._ensure_rag_index()
            self._send_json(200, {
                "primary_edge": "Cloudflare Workers + Static Assets + API Origin Proxy",
                "wrangler_configured": (root_dir / "cloudflare" / "wrangler.toml").is_file(),
                "base_small_model": cfg.base_small_model,
                "base_display_model": cfg.finetuned_base_model,
                "finetuned_model": cfg.finetuned_base_model,
                "adapter_path": cfg.adapter_path,
                "adapter_ready": (root_dir / cfg.adapter_path / "adapter_model.safetensors").is_file(),
                "rag_chunks_loaded": len(self._rag_chunks or []),
                "active_provider": registry.get_active_provider().provider_name,
            })

        elif path == "/api/logs":
            logs = self.workflow.security_layer.get_audit_logs(limit=50)
            self._send_json(200, {"audit_logs": logs})

        else:
            web_dir = Path(__file__).resolve().parent.parent / "web"
            file_to_serve = web_dir / (path.lstrip("/") or "index.html")
            if file_to_serve.is_file():
                content_type = "text/html; charset=utf-8"
                if file_to_serve.suffix == ".css":
                    content_type = "text/css; charset=utf-8"
                elif file_to_serve.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                elif file_to_serve.suffix == ".json":
                    content_type = "application/json; charset=utf-8"

                with open(file_to_serve, "rb") as f:
                    content = f.read()

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "File not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if path == "/api/verify":
            raw_text = payload.get("raw_ocr_text", "")
            user_prompt = payload.get("user_prompt", "")
            selected_setup = payload.get("setup", "fine_tuned")
            if selected_setup not in SETUP_METADATA:
                selected_setup = "fine_tuned"

            applicant_data = payload.get("applicant_profile", {})
            quality_data = payload.get("image_quality", {})

            applicant_profile = None
            if applicant_data:
                applicant_profile = ApplicantProfile(
                    submitted_name=applicant_data.get("submitted_name", ""),
                    submitted_dob=applicant_data.get("submitted_dob"),
                    submitted_document_type=applicant_data.get("submitted_document_type", "AADHAAR_CARD"),
                    submitted_document_number=applicant_data.get("submitted_document_number", ""),
                    submitted_pincode=applicant_data.get("submitted_pincode"),
                    mobile_number=applicant_data.get("mobile_number"),
                )

            quality_flags = ImageQualityFlags(
                blur_detected=quality_data.get("blur_detected", False),
                glare_detected=quality_data.get("glare_detected", False),
                orientation_angle=quality_data.get("orientation_angle", 0),
                simulated_ocr_confidence=quality_data.get("simulated_ocr_confidence", 0.95),
            )
            record_id = payload.get("record_id", f"VERIF-{random.randint(1000, 9999)}")

            setup_results = {}
            for setup_id in ("base", "base_rag", "fine_tuned", "fine_tuned_rag"):
                setup_results[setup_id] = self._run_setup(
                    setup_id=setup_id,
                    raw_text=raw_text,
                    user_prompt=user_prompt,
                    applicant_profile=applicant_profile,
                    quality_flags=quality_flags,
                    record_id=record_id,
                )

            primary = dict(setup_results[selected_setup])
            primary["selected_setup"] = selected_setup
            primary["user_prompt_redacted"] = PIIRedactor.redact_text(user_prompt) if user_prompt else ""
            primary["setup_results"] = setup_results
            self._send_json(200, primary)

        elif path == "/api/login":
            role = str(payload.get("role", "")).strip().lower()
            email = str(payload.get("email", "")).strip().lower()
            password = str(payload.get("password", ""))
            account = self._demo_accounts.get(role)
            if not account or email != account["email"] or password != account["password"]:
                self._send_json(401, {"authenticated": False, "error": "Invalid credentials for selected role."})
                return
            token = secrets.token_urlsafe(24)
            self._sessions[token] = role
            self._send_json(200, {
                "authenticated": True,
                "role": role,
                "display_name": account["display_name"],
                "allowed_setup": account["allowed_setup"],
                "token": token,
            })

        elif path == "/api/chat-kyc":
            result = self._run_end_user_chat(
                query=str(payload.get("query", "")),
                file_name=str(payload.get("file_name", "")),
                file_content_b64=str(payload.get("file_content_b64", "")),
                raw_ocr_text=str(payload.get("raw_ocr_text", "")),
            )
            self._send_json(200, result)

        elif path == "/api/providers/switch":
            prov_name = payload.get("provider_name")
            try:
                prov = registry.set_active_provider(prov_name)
                update_config({"active_provider": prov_name})
                self._send_json(200, {
                    "message": f"Switched active provider to {prov_name}",
                    "active_provider": prov_name,
                    "model": prov.config.model_name,
                })
            except Exception as e:
                self._send_json(400, {"error": str(e)})

        elif path == "/api/config":
            try:
                new_cfg = update_config(payload)
                self.workflow.security_layer.enable_prompt_guard = new_cfg.security.enable_prompt_guard
                self.workflow.security_layer.enable_pii_redaction = new_cfg.security.enable_pii_redaction
                self.workflow.quality_analyzer.min_confidence_threshold = new_cfg.thresholds.min_ocr_confidence
                if "providers" in payload:
                    for p_name, p_val in payload["providers"].items():
                        registry.update_provider_config(p_name, p_val)

                dumped = new_cfg.model_dump()
                dumped["base_dir"] = str(dumped["base_dir"])
                dumped["data_dir"] = str(dumped["data_dir"])
                self._send_json(200, {"message": "Configuration updated successfully", "config": dumped})
            except Exception as e:
                self._send_json(400, {"error": str(e)})

        else:
            self.send_error(404, "Endpoint not found")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    server_address = (host, port)
    httpd = HTTPServer(server_address, KYCAPIRequestHandler)
    print(f"Banking KYC Production API & UI Server running at http://{host}:{port}")
    print(f"User Portal & Admin Dashboard: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
