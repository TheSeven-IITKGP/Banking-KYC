"""Data models for Banking KYC Verification System.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class KYCDocumentType(str, Enum):
    AADHAAR_CARD = "AADHAAR_CARD"
    PAN_CARD = "PAN_CARD"
    VOTER_ID = "VOTER_ID"
    DRIVING_LICENCE = "DRIVING_LICENCE"
    PASSPORT = "PASSPORT"
    UNKNOWN = "UNKNOWN"


class ScenarioTag(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    FUZZY_MATCH_OCR_NOISE = "FUZZY_MATCH_OCR_NOISE"
    MISMATCH_NAME = "MISMATCH_NAME"
    FUZZY_MATCH_FORMAT_DIFF = "FUZZY_MATCH_FORMAT_DIFF"
    MISMATCH_DOB = "MISMATCH_DOB"
    REJECT_BLUR_GLARE = "REJECT_BLUR_GLARE"
    MISMATCH_ID_FORMAT = "MISMATCH_ID_FORMAT"


class VerificationDecision(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    APPROVED_WITH_FUZZY_MATCH = "APPROVED_WITH_FUZZY_MATCH"
    MANUAL_REVIEW_OR_REJECT = "MANUAL_REVIEW_OR_REJECT"
    REJECTED_SECURITY_VIOLATION = "REJECTED_SECURITY_VIOLATION"
    REJECTED_POOR_QUALITY = "REJECTED_POOR_QUALITY"


class ProviderType(str, Enum):
    LOCAL_FINE_TUNED = "LOCAL_FINE_TUNED"
    VLLM = "VLLM"
    OLLAMA = "OLLAMA"
    HUGGINGFACE = "HUGGINGFACE"


class Address(BaseModel):
    line1: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None


class ExtractedKYCData(BaseModel):
    document_type: KYCDocumentType = KYCDocumentType.UNKNOWN
    document_number: Optional[str] = None
    full_name: Optional[str] = None
    father_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[Address] = None
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0


class ApplicantProfile(BaseModel):
    submitted_name: str
    submitted_dob: Optional[str] = None
    submitted_document_type: str = "AADHAAR_CARD"
    submitted_document_number: str
    submitted_pincode: Optional[str] = None
    mobile_number: Optional[str] = None


class ImageQualityFlags(BaseModel):
    blur_detected: bool = False
    glare_detected: bool = False
    orientation_angle: int = 0
    simulated_ocr_confidence: float = 0.95


class FieldComparison(BaseModel):
    document_number_exact_match: bool = False
    name_similarity_score: float = 0.0
    dob_match: bool = False
    pincode_match: Optional[bool] = None


class SecurityCheckResult(BaseModel):
    is_safe: bool = True
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    threat_type: Optional[str] = None
    details: str = "Clean"
    sanitized_input: Optional[str] = None


class ProviderConfig(BaseModel):
    provider_type: ProviderType = ProviderType.LOCAL_FINE_TUNED
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct-KYC"
    base_url: Optional[str] = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 30


class VerificationResult(BaseModel):
    record_id: str = "VERIF-LIVE-001"
    overall_match: bool
    decision: VerificationDecision
    confidence_score: float = 0.0
    field_comparison: FieldComparison
    extracted_data: ExtractedKYCData
    applicant_profile: ApplicantProfile
    sanitized_view: Dict[str, Any] = Field(default_factory=dict)
    security_check: SecurityCheckResult = Field(default_factory=SecurityCheckResult)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    active_provider: str = ProviderType.LOCAL_FINE_TUNED.value
