"""LangGraph State Graph Implementation for Banking KYC Verification Pipeline.

Orchestrates sequential and conditional execution through Inbound Security,
OCR Ingestion, Model Extraction, Multi-Field Verification, Decision Policy,
and Outbound PII Redaction.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from banking_kyc.config import get_config
from banking_kyc.models import (
    ApplicantProfile,
    ExtractedKYCData,
    FieldComparison,
    ImageQualityFlags,
    SecurityCheckResult,
    VerificationDecision,
    VerificationResult,
)
from banking_kyc.ocr.engine import HeuristicRegexExtractor
from banking_kyc.ocr.normalizer import DocumentNormalizer
from banking_kyc.ocr.preprocessor import ImageQualityAnalyzer
from banking_kyc.providers.registry import registry
from banking_kyc.security.guardrail import SecurityLayer


class KYCWorkflowState(BaseModel):
    """Execution state container passed through LangGraph nodes."""

    record_id: str = "LIVE-001"
    raw_ocr_text: str = ""
    applicant_profile: Optional[ApplicantProfile] = None
    image_quality: ImageQualityFlags = Field(default_factory=ImageQualityFlags)
    security_check: SecurityCheckResult = Field(default_factory=SecurityCheckResult)
    extracted_data: Optional[ExtractedKYCData] = None
    field_comparison: FieldComparison = Field(default_factory=FieldComparison)
    overall_match: bool = False
    decision: VerificationDecision = VerificationDecision.MANUAL_REVIEW_OR_REJECT
    confidence_score: float = 0.0
    sanitized_view: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None


class LangGraphKYCWorkflow:
    """Production StateGraph verifying KYC records."""

    def __init__(self):
        self.config = get_config()
        self.security_layer = SecurityLayer(
            enable_prompt_guard=self.config.security.enable_prompt_guard,
            enable_pii_redaction=self.config.security.enable_pii_redaction,
        )
        self.quality_analyzer = ImageQualityAnalyzer(
            min_confidence_threshold=self.config.thresholds.min_ocr_confidence
        )

    # ==================== NODE DEFINITIONS ====================

    def inbound_security_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        res = self.security_layer.inspect_inbound(state.raw_ocr_text)
        state.security_check = res
        state.execution_trace.append({
            "node": "inbound_security_guard",
            "is_safe": res.is_safe,
            "risk_level": res.risk_level,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })
        return state

    def ocr_quality_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        ok, flags, msg = self.quality_analyzer.evaluate_quality(
            raw_text=state.raw_ocr_text,
            simulated_confidence=state.image_quality.simulated_ocr_confidence,
            blur_detected=state.image_quality.blur_detected,
            glare_detected=state.image_quality.glare_detected,
            orientation_angle=state.image_quality.orientation_angle,
        )
        state.image_quality = flags
        state.execution_trace.append({
            "node": "ocr_quality_analyzer",
            "quality_ok": ok,
            "reason": msg,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })
        if not ok:
            state.error_message = msg
        return state

    def extraction_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        provider = registry.get_active_provider()
        extracted = provider.extract_kyc_data(state.raw_ocr_text)
        state.extracted_data = extracted
        state.execution_trace.append({
            "node": "model_extraction",
            "active_provider": provider.provider_name,
            "document_type": extracted.document_type.value,
            "has_document_number": bool(extracted.document_number),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })
        return state

    def field_matching_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        prof = state.applicant_profile
        ext = state.extracted_data

        if not prof or not ext:
            state.execution_trace.append({
                "node": "field_matching",
                "status": "SKIPPED_MISSING_INPUTS",
            })
            return state

        # Document Number Matching
        norm_ext_doc = DocumentNormalizer.normalize_aadhaar(ext.document_number) if "AADHAAR" in ext.document_type.value else (ext.document_number or "").strip().upper()
        norm_sub_doc = DocumentNormalizer.normalize_aadhaar(prof.submitted_document_number) if "AADHAAR" in prof.submitted_document_type.upper() else (prof.submitted_document_number or "").strip().upper()
        doc_num_match = (norm_ext_doc == norm_sub_doc) and bool(norm_ext_doc)

        # Name Matching
        norm_ext_name = DocumentNormalizer.normalize_name(ext.full_name)
        norm_sub_name = DocumentNormalizer.normalize_name(prof.submitted_name)
        sim_score = DocumentNormalizer.calculate_levenshtein_ratio(norm_ext_name, norm_sub_name)

        # DOB Matching
        norm_ext_dob = DocumentNormalizer.normalize_date(ext.date_of_birth)
        norm_sub_dob = DocumentNormalizer.normalize_date(prof.submitted_dob)
        dob_match = (norm_ext_dob == norm_sub_dob) and bool(norm_ext_dob)

        # Pincode Matching
        pincode_match = None
        if ext.address and ext.address.pincode and prof.submitted_pincode:
            pincode_match = (ext.address.pincode.strip() == prof.submitted_pincode.strip())

        state.field_comparison = FieldComparison(
            document_number_exact_match=doc_num_match,
            name_similarity_score=round(sim_score, 3),
            dob_match=dob_match,
            pincode_match=pincode_match,
        )

        state.execution_trace.append({
            "node": "field_matching",
            "doc_num_match": doc_num_match,
            "name_similarity": round(sim_score, 3),
            "dob_match": dob_match,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })
        return state

    def decision_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        fc = state.field_comparison
        thresh = self.config.thresholds

        # Policy Logic:
        # 1. Document number must match exactly
        # 2. DOB must match
        # 3. Name similarity controls approval vs fuzzy vs reject
        if not fc.document_number_exact_match:
            state.overall_match = False
            state.decision = VerificationDecision.MANUAL_REVIEW_OR_REJECT
            state.confidence_score = 0.30
        elif thresh.strict_dob_match and not fc.dob_match:
            state.overall_match = False
            state.decision = VerificationDecision.MANUAL_REVIEW_OR_REJECT
            state.confidence_score = 0.45
        elif fc.name_similarity_score >= 0.95:
            state.overall_match = True
            state.decision = VerificationDecision.AUTO_APPROVED
            state.confidence_score = 0.98
        elif fc.name_similarity_score >= thresh.fuzzy_approval_threshold:
            state.overall_match = True
            state.decision = VerificationDecision.APPROVED_WITH_FUZZY_MATCH
            state.confidence_score = 0.88
        else:
            state.overall_match = False
            state.decision = VerificationDecision.MANUAL_REVIEW_OR_REJECT
            state.confidence_score = 0.50

        state.execution_trace.append({
            "node": "decision_engine",
            "overall_match": state.overall_match,
            "decision": state.decision.value,
            "confidence_score": state.confidence_score,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        })
        return state

    def security_rejection_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        state.overall_match = False
        state.decision = VerificationDecision.REJECTED_SECURITY_VIOLATION
        state.confidence_score = 0.0
        state.execution_trace.append({
            "node": "security_rejection_handler",
            "threat": state.security_check.threat_type,
            "action": "HALT_AND_REJECT",
        })
        return state

    def quality_rejection_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        state.overall_match = False
        state.decision = VerificationDecision.REJECTED_POOR_QUALITY
        state.confidence_score = 0.0
        state.execution_trace.append({
            "node": "quality_rejection_handler",
            "reason": state.error_message,
            "action": "REQUEST_RESCAN",
        })
        return state

    def outbound_redaction_node(self, state: KYCWorkflowState) -> KYCWorkflowState:
        t0 = time.time()
        # Build sanitized customer / dashboard view
        raw_view: Dict[str, Any] = {
            "record_id": state.record_id,
            "decision": state.decision.value,
            "overall_match": state.overall_match,
            "confidence_score": state.confidence_score,
            "extracted_document_type": state.extracted_data.document_type.value if state.extracted_data else None,
            "extracted_document_number": state.extracted_data.document_number if state.extracted_data else None,
            "extracted_full_name": state.extracted_data.full_name if state.extracted_data else None,
            "extracted_dob": state.extracted_data.date_of_birth if state.extracted_data else None,
            "applicant_name": state.applicant_profile.submitted_name if state.applicant_profile else None,
            "applicant_document_number": state.applicant_profile.submitted_document_number if state.applicant_profile else None,
            "applicant_mobile": state.applicant_profile.mobile_number if state.applicant_profile else None,
            "field_comparison": state.field_comparison.model_dump(),
        }

        state.sanitized_view = self.security_layer.sanitize_outbound_dict(raw_view)
        state.end_time = time.time()
        state.execution_trace.append({
            "node": "outbound_pii_redaction",
            "pii_masked": self.config.security.enable_pii_redaction,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "total_workflow_time_ms": round((state.end_time - state.start_time) * 1000, 2),
        })
        return state

    # ==================== PIPELINE EXECUTION ====================

    def execute(self, state: KYCWorkflowState) -> VerificationResult:
        """Executes the StateGraph transitions deterministically."""
        # Step 1: Inbound Security Guard
        state = self.inbound_security_node(state)
        if not state.security_check.is_safe:
            state = self.security_rejection_node(state)
            state = self.outbound_redaction_node(state)
            return self._build_result(state)

        # Step 2: OCR Quality Analyzer
        state = self.ocr_quality_node(state)
        if state.error_message:
            state = self.quality_rejection_node(state)
            state = self.outbound_redaction_node(state)
            return self._build_result(state)

        # Step 3: Model Extraction
        state = self.extraction_node(state)

        # Step 4: Multi-Field Comparison
        state = self.field_matching_node(state)

        # Step 5: Decision Policy
        state = self.decision_node(state)

        # Step 6: Outbound PII Redaction
        state = self.outbound_redaction_node(state)

        return self._build_result(state)

    def _build_result(self, state: KYCWorkflowState) -> VerificationResult:
        total_time = (state.end_time - state.start_time) * 1000 if state.end_time else 0.0
        active_prov = registry.get_active_provider().provider_name
        return VerificationResult(
            record_id=state.record_id,
            overall_match=state.overall_match,
            decision=state.decision,
            confidence_score=state.confidence_score,
            field_comparison=state.field_comparison,
            extracted_data=state.extracted_data or ExtractedKYCData(),
            applicant_profile=state.applicant_profile or ApplicantProfile(submitted_name="", submitted_document_number=""),
            sanitized_view=state.sanitized_view,
            security_check=state.security_check,
            execution_trace=state.execution_trace,
            processing_time_ms=round(total_time, 2),
            active_provider=active_prov,
        )
