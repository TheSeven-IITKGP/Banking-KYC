"""Production HTTP REST API Server and Web UI Handler.

Implements REST endpoints for KYC verification, dataset sample fetching,
dynamic admin configuration, provider hot-swapping, and metrics monitoring.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import random
import sys
import urllib.parse
from typing import Any, Dict

from banking_kyc.config import get_config, update_config
from banking_kyc.finetuning.evaluator import KYCEvaluator
from banking_kyc.models import (
    ApplicantProfile,
    ImageQualityFlags,
    ProviderType,
)
from banking_kyc.providers.registry import registry
from banking_kyc.workflow.graph import KYCWorkflowState, LangGraphKYCWorkflow


class KYCAPIRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request handler supporting KYC REST endpoints and Web UI."""

    workflow: LangGraphKYCWorkflow = LangGraphKYCWorkflow()
    dataset_records = None

    @classmethod
    def _get_dataset_sample(cls, scenario: str | None = None) -> Dict[str, Any]:
        if cls.dataset_records is None:
            cls.dataset_records = []
            jsonl_path = Path(__file__).resolve().parent.parent / "synthetic_indian_kyc_ocr_finetune_2000.jsonl"
            if not jsonl_path.exists():
                jsonl_path = Path("Banking-KYC/synthetic_indian_kyc_ocr_finetune_2000.jsonl")
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            cls.dataset_records.append(json.loads(line))

        if not cls.dataset_records:
            return {}

        candidates = cls.dataset_records
        if scenario:
            candidates = [r for r in cls.dataset_records if r.get("scenario_tag") == scenario]
            if not candidates:
                candidates = cls.dataset_records

        return random.choice(candidates)

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
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

        if path == "/api/config":
            cfg = get_config().model_dump()
            # Convert Path objects to str
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
            self._send_json(200, {"benchmark_report": benchmark})

        elif path == "/api/logs":
            logs = self.workflow.security_layer.get_audit_logs(limit=50)
            self._send_json(200, {"audit_logs": logs})

        else:
            # Serve web UI assets
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

            state = KYCWorkflowState(
                record_id=payload.get("record_id", f"VERIF-{random.randint(1000, 9999)}"),
                raw_ocr_text=raw_text,
                applicant_profile=applicant_profile,
                image_quality=quality_flags,
            )

            result = self.workflow.execute(state)
            self._send_json(200, result.model_dump())

        elif path == "/api/providers/switch":
            prov_name = payload.get("provider_name")
            try:
                prov = registry.set_active_provider(prov_name)
                # Update config
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
                # Sync security layer
                self.workflow.security_layer.enable_prompt_guard = new_cfg.security.enable_prompt_guard
                self.workflow.security_layer.enable_pii_redaction = new_cfg.security.enable_pii_redaction
                self.workflow.quality_analyzer.min_confidence_threshold = new_cfg.thresholds.min_ocr_confidence
                # Sync provider configs
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
    print(f"🚀 Banking KYC Production API & UI Server running at http://{host}:{port}")
    print(f"👉 User Portal & Admin Dashboard: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
