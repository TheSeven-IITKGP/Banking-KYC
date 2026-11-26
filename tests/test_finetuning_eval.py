from pathlib import Path
import tokenize
import io
import unittest

from banking_kyc.api.server import KYCAPIRequestHandler
from banking_kyc.finetuning.dataset_prep import KYCDatasetPreparer
from banking_kyc.finetuning.evaluator import KYCEvaluator
from banking_kyc.finetuning.qlora_config import QLoRAExperimentConfigs
from banking_kyc.models import ApplicantProfile, ImageQualityFlags


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

    def test_combined_four_setup_execution_and_rag(self):
        ocr_text = (
            "GOVERNMENT OF INDIA\n"
            "Unique Identification Authority of India\n"
            "ROHAN DESHMUKH\n"
            "DOB: 21-12-1967\n"
            "MALE\n"
            "Address:\n"
            "Flat 836, Tower C, Anna Nagar,\n"
            "New Delhi, Delhi - 110082\n"
            "XXXX-XXXX-5339"
        )
        profile = ApplicantProfile(
            submitted_name="Rohan Deshmukh",
            submitted_dob="1967-12-21",
            submitted_document_type="AADHAAR_CARD",
            submitted_document_number="XXXX-XXXX-5339",
            submitted_pincode="110082",
            mobile_number="+91-9647099690",
        )
        qflags = ImageQualityFlags(blur_detected=False, glare_detected=False, simulated_ocr_confidence=0.95)
        results = {}
        for sid in ("base", "base_rag", "fine_tuned", "fine_tuned_rag"):
            results[sid] = KYCAPIRequestHandler._run_setup(
                setup_id=sid,
                raw_text=ocr_text,
                user_prompt="Verify Aadhaar document and cite RBI officially valid document rules",
                applicant_profile=profile,
                quality_flags=qflags,
                record_id="TEST-COMBINED-001",
            )
            self.assertTrue(results[sid]["ready"])
            self.assertTrue(results[sid]["realtime_eval"]["ready"])

        self.assertEqual(results["base"]["rag_citations"], [])
        self.assertEqual(results["fine_tuned"]["rag_citations"], [])
        self.assertGreater(len(results["base_rag"]["rag_citations"]), 0)
        self.assertGreater(len(results["fine_tuned_rag"]["rag_citations"]), 0)
        self.assertIn("Qwen2.5-7B-Instruct", results["base"]["model_used"])
        self.assertIn("Qwen2.5-7B-Instruct", results["base_rag"]["model_used"])
        self.assertIn("1.5B", results["base"]["runtime_model"])
        self.assertIn("qwen2.5-7b-qlora", results["fine_tuned_rag"]["model_used"])

    def test_streamlit_app_and_finetuning_ui(self):
        root = Path(__file__).resolve().parent.parent
        ft_app_path = root / "app" / "finetuning_streamlit_app.py"
        unified_app_path = root / "app" / "streamlit_app.py"
        self.assertTrue(ft_app_path.exists())
        self.assertTrue(unified_app_path.exists())

        content = unified_app_path.read_text(encoding="utf-8")
        ft_content = ft_app_path.read_text(encoding="utf-8")
        web_css = (root / "banking_kyc" / "web" / "styles.css").read_text(encoding="utf-8")
        web_html = (root / "banking_kyc" / "web" / "index.html").read_text(encoding="utf-8")
        web_js = (root / "banking_kyc" / "web" / "app.js").read_text(encoding="utf-8")
        streamlit_cfg = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")

        self.assertIn("FT_STATUS_LABEL", content)
        self.assertIn("RAG_STATUS_LABEL", content)
        self.assertIn('"green"', content)
        self.assertIn('"orange"', content)
        self.assertIn('"red"', content)
        self.assertIn('"gray"', content)
        self.assertIn("KYC document verification", content)
        self.assertIn("KYC rules assistant", content)
        self.assertIn("Sources cited", content)
        self.assertIn("Sources cited", ft_content)
        self.assertIn("answer_text", ft_content)
        self.assertIn("status-pill", content)
        self.assertIn("status-pill", ft_content)
        self.assertIn("#132a1b", content)
        self.assertIn("#132a1b", ft_content)
        self.assertIn("#bbf7d0", content)
        self.assertIn("#bbf7d0", ft_content)
        self.assertIn("warn-box", content)
        self.assertIn("warn-box", ft_content)
        self.assertIn('base = "dark"', streamlit_cfg)
        self.assertIn('backgroundColor = "#121212"', streamlit_cfg)
        self.assertIn('name="color-scheme" content="dark"', web_html)

        for ui_id in (
            "roleSelectDialog",
            "chooseUserRoleBtn",
            "chooseAdminRoleBtn",
            "loginSection",
            "userLoginForm",
            "adminLoginForm",
            "dummyCredsBox",
            "dialogDummyCreds",
            "userChatSection",
            "kycDocUpload",
            "chatForm",
            "adminWorkspace",
            "setupSelect",
            "realtimeEvalTrigger",
            "triggerSetupBase",
            "triggerSetupBaseRag",
            "triggerSetupFt",
            "triggerSetupFtRag",
            "realtimeEvalDialog",
            "setupResultDialog",
            "modelCardSection",
            "benchSection",
            "adminSection",
        ):
            self.assertIn(ui_id, web_html)

        for token in ("#132a1b", "#bbf7d0", "#166534", "#2b1d0b", "#fde68a", "#92400e", "#2c1215", "#fecaca", "#991b1b", "#28220c", "#fef08a", "#262626", "#444444", "#121212", "#f0f0f0"):
            self.assertIn(token, web_css)

        legacy_tokens = ("#1e3a8a", "#2563eb", "#0f172a", "#38bdf8", "#a5f3fc", "#f8fafc", "#f1f5f9", "#e2e8f0", "#eff6ff")
        for asset in (web_css, web_html, web_js, content, ft_content, streamlit_cfg):
            for legacy in legacy_tokens:
                self.assertNotIn(legacy, asset.lower())

        for asset in (web_html, web_js, content, ft_content):
            self.assertNotIn("Lead Engineer", asset)
            for ch in asset:
                self.assertFalse(
                    0x1F300 <= ord(ch) <= 0x1FAFF or ord(ch) == 0xFE0F or ch in "⚠️⚡⏳⚙️",
                    f"Unexpected emoji icon {ch!r} in frontend asset",
                )

        self.assertTrue((root / "cloudflare" / "wrangler.toml").is_file())
        self.assertTrue((root / "cloudflare" / "worker.mjs").is_file())
        self.assertNotIn("1.5B", web_html)
        self.assertNotIn("1.5B", web_js)
        self.assertTrue((root / "MODEL_CARD.md").is_file())
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", (root / "MODEL_CARD.md").read_text(encoding="utf-8"))
        self.assertNotIn("[More Information Needed]", (root / "models" / "adapters" / "qwen2.5-7b-qlora" / "README.md").read_text(encoding="utf-8"))

        docs_dir = root.parent / "documentation"
        self.assertTrue((docs_dir / "DEPLOYMENT_PLATFORMS.md").is_file())
        self.assertTrue((docs_dir / "VIDEO_DEMO_SCRIPT.md").is_file())
        self.assertTrue((docs_dir / "CODEBASE_REFERENCE.md").is_file())

        py_targets = list(root.glob("*.py"))
        for py_root in ("banking_kyc", "app", "src", "tests", "eval", "deploy"):
            py_targets.extend((root / py_root).rglob("*.py"))
        for py_file in py_targets:
            toks = list(tokenize.generate_tokens(io.StringIO(py_file.read_text(encoding="utf-8")).readline))
            comments = [t for t in toks if t.type == tokenize.COMMENT and not (t.start[0] == 1 and t.string.startswith("#!"))]
            self.assertEqual(comments, [], f"Found comments in {py_file}: {comments}")

    def test_dataset_sample_loading_and_cache_headers(self):
        from banking_kyc.api.server import KYCAPIRequestHandler

        KYCAPIRequestHandler.dataset_records = []
        exact = KYCAPIRequestHandler._get_dataset_sample("EXACT_MATCH")
        self.assertEqual(exact.get("scenario_tag"), "EXACT_MATCH")
        self.assertIn("ocr_input", exact)
        self.assertIn("raw_ocr_text", exact["ocr_input"])

        rnd = KYCAPIRequestHandler._get_dataset_sample()
        self.assertIn("ocr_input", rnd)
        self.assertIn("applicant_submitted_profile", rnd)

        root = Path(__file__).resolve().parent.parent
        server_src = (root / "banking_kyc" / "api" / "server.py").read_text(encoding="utf-8")
        app_js = (root / "banking_kyc" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('"Cache-Control", "no-store"', server_src)
        self.assertIn('cache: "no-store"', app_js)

    def test_docker_and_gpu_deployment_artifacts(self):
        import ast
        import os
        from banking_kyc.api.server import KYCAPIRequestHandler

        root = Path(__file__).resolve().parent.parent
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        modal_src = (root / "deploy" / "modal_app.py").read_text(encoding="utf-8")

        self.assertIn(".venv/", dockerignore)
        self.assertIn(".git/", dockerignore)
        self.assertIn("PYTHONPATH=/app:/app/src", dockerfile)
        self.assertIn("PORT=8080", dockerfile)
        self.assertIn("/api/health", dockerfile)

        ast.parse(modal_src)
        self.assertIn("FINKYC_LIVE_GPU", modal_src)
        self.assertIn('gpu="L4"', modal_src)
        self.assertIn("models/adapters/qwen2.5-7b-qlora", modal_src)

        old_flag = os.environ.pop("FINKYC_LIVE_GPU", None)
        try:
            KYCAPIRequestHandler._setup_providers.clear()
            self.assertIsNone(KYCAPIRequestHandler._get_setup_provider("base"))
            os.environ["FINKYC_LIVE_GPU"] = "1"
            base_prov = KYCAPIRequestHandler._get_setup_provider("base")
            ft_prov = KYCAPIRequestHandler._get_setup_provider("fine_tuned")
            self.assertIsNotNone(base_prov)
            self.assertIsNotNone(ft_prov)
            self.assertIn("1.5B", base_prov.config.model_name)
            self.assertIsNone(base_prov.config.adapter_path)
            self.assertIn("7B", ft_prov.config.model_name)
            self.assertIn("qwen2.5-7b-qlora", ft_prov.config.adapter_path or "")
        finally:
            KYCAPIRequestHandler._setup_providers.clear()
            if old_flag is None:
                os.environ.pop("FINKYC_LIVE_GPU", None)
            else:
                os.environ["FINKYC_LIVE_GPU"] = old_flag

    def test_login_personas_and_user_chat_finetuned_rag_only(self):
        import io as sys_io
        import json
        from banking_kyc.api.server import KYCAPIRequestHandler

        class FakeSocket:
            def __init__(self, raw_request: bytes):
                self.rfile = sys_io.BytesIO(raw_request)
                self.wfile = sys_io.BytesIO()

            def makefile(self, mode, *args, **kwargs):
                return self.rfile if "r" in mode else self.wfile

            def sendall(self, data: bytes):
                self.wfile.write(data)

        def call_post(path: str, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            req = (
                f"POST {path} HTTP/1.1\r\n"
                f"Host: localhost\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode("utf-8") + body
            sock = FakeSocket(req)
            KYCAPIRequestHandler(sock, ("127.0.0.1", 0), None)
            raw_resp = sock.wfile.getvalue().decode("utf-8")
            head, resp_body = raw_resp.split("\r\n\r\n", 1)
            status_code = int(head.splitlines()[0].split()[1])
            return status_code, json.loads(resp_body)

        code_ok_user, user_res = call_post("/api/login", {"role": "user", "email": "user@finkyc.in", "password": "User@1234"})
        self.assertEqual(code_ok_user, 200)
        self.assertTrue(user_res["authenticated"])
        self.assertEqual(user_res["allowed_setup"], "fine_tuned_rag")

        code_ok_admin, admin_res = call_post("/api/login", {"role": "admin", "email": "admin@finkyc.in", "password": "Admin@1234"})
        self.assertEqual(code_ok_admin, 200)
        self.assertEqual(admin_res["role"], "admin")

        code_bad, _ = call_post("/api/login", {"role": "admin", "email": "user@finkyc.in", "password": "wrong"})
        self.assertEqual(code_bad, 401)

        chat_code, chat_res = call_post("/api/chat-kyc", {
            "query": "Verify this Aadhaar card and cite RBI officially valid document rules",
            "file_name": "sample.txt",
            "raw_ocr_text": "GOVERNMENT OF INDIA\nUnique Identification Authority of India\nANIL MENON\nDOB: 24-03-1972\nMALE\nXXXX-XXXX-4822",
            "setup": "base",
        })
        self.assertEqual(chat_code, 200)
        self.assertEqual(chat_res["setup"], "fine_tuned_rag")
        self.assertIn("qwen2.5-7b-qlora", chat_res["model_used"])
        self.assertGreater(len(chat_res["rag_citations"]), 0)
        self.assertEqual(chat_res["sanitized_view"]["document_number"], "XXXX-XXXX-4822")
        self.assertEqual(chat_res["status_label"], "Document Checked")
        self.assertGreater(len(chat_res["document_details"]), 0)

        rag_code, rag_res = call_post("/api/chat-kyc", {"query": "What is a KYC Identifier and what documents are required for opening a bank account?"})
        self.assertEqual(rag_code, 200)
        self.assertEqual(rag_res["setup"], "fine_tuned_rag")
        self.assertGreater(len(rag_res["rag_citations"]), 0)
        self.assertGreater(len(rag_res["key_points"]), 0)
        longest_src = max(len(c["text"]) for c in rag_res["rag_citations"])
        combined_kp = " ".join(p["body"] for p in rag_res["key_points"])
        self.assertGreaterEqual(len(combined_kp), min(longest_src - 60, 260))
        self.assertNotIn("equiva |", rag_res["answer"])

    def test_image_ocr_upload_and_aadhaar_parsing(self):
        import base64
        import io as sys_io
        import json
        from unittest.mock import patch
        from PIL import Image
        from banking_kyc.api.server import KYCAPIRequestHandler
        from banking_kyc.ocr.engine import HeuristicRegexExtractor, extract_text_from_image_bytes

        sample_ocr = (
            "Mera Aadhaar, Meri Pehchan\n"
            "GOVERNMENT OF INDIA\n"
            "Unique Identification Authority of India\n"
            "CHANDAN BAURI\n"
            "DOB: 15/08/1996\n"
            "MALE\n"
            "VID : 9111 2222 3333 4444\n"
            "1234 5678 9012\n"
            "PIN 700001"
        )
        parsed = HeuristicRegexExtractor.extract(sample_ocr)
        self.assertEqual(parsed.document_type.value, "AADHAAR_CARD")
        self.assertEqual(parsed.document_number, "1234-5678-9012")
        self.assertEqual(parsed.full_name, "CHANDAN BAURI")
        self.assertEqual(parsed.date_of_birth, "1996-08-15")
        self.assertEqual(parsed.gender, "MALE")

        buf = sys_io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with patch("banking_kyc.ocr.engine._ensure_darwin_vision_binary", return_value=None), \
             patch("banking_kyc.ocr.engine.shutil.which", return_value="/usr/bin/tesseract"), \
             patch("banking_kyc.ocr.engine.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = sample_ocr
            extracted_txt = extract_text_from_image_bytes(jpeg_bytes, "aadhaar_photo.jpeg")
            self.assertIn("CHANDAN BAURI", extracted_txt)
            self.assertIn("1234 5678 9012", extracted_txt)

        with patch("banking_kyc.api.server.extract_text_from_image_bytes", return_value=sample_ocr):
            res = KYCAPIRequestHandler._run_end_user_chat(
                query="",
                file_name="WhatsApp Image.jpeg",
                file_content_b64=base64.b64encode(jpeg_bytes).decode("ascii"),
            )
            self.assertEqual(res["setup"], "fine_tuned_rag")
            self.assertEqual(res["decision"], "DOCUMENT_VERIFIED_WITH_RAG")
            self.assertEqual(res["sanitized_view"]["document_number"], "XXXX-XXXX-9012")
            self.assertGreater(len(res["rag_citations"]), 0)
            self.assertNotIn("FAQ 17", res["answer"])


if __name__ == "__main__":
    unittest.main()
