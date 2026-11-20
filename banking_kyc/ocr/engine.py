"""OCR Engines for Indian KYC Ingestion and Extraction.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import abc
import re
from typing import Any, Dict, Optional

from banking_kyc.models import Address, ExtractedKYCData, KYCDocumentType
from banking_kyc.ocr.normalizer import DocumentNormalizer


class BaseOCREngine(abc.ABC):
    """Abstract base class for OCR extraction engines."""

    @abc.abstractmethod
    def extract_text(self, document_source: Any) -> str:
        """Extracts raw text from an image, pdf, or simulated buffer."""
        pass

    @abc.abstractmethod
    def parse_kyc_fields(self, raw_text: str) -> ExtractedKYCData:
        """Parses structured KYC fields from the raw OCR text."""
        pass


class HeuristicRegexExtractor:
    """High-precision pattern matcher for Indian Government ID formats.

    Covers Aadhaar, PAN Card, Passport, Voter ID, and Driving Licence.
    """

    # RegEx Patterns
    AADHAAR_PATTERN = re.compile(r"\b(XXXX[- ]?XXXX[- ]?\d{4}|\d{4}[- ]\d{4}[- ]\d{4})\b", re.IGNORECASE)
    PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    PASSPORT_PATTERN = re.compile(r"\b([A-Z][0-9]{7})\b")
    VOTER_ID_PATTERN = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")
    DL_PATTERN = re.compile(r"\b([A-Z]{2}[0-9]{2}\s?[0-9]{11})\b")

    DOB_PATTERN = re.compile(r"(?:DOB|Date of Birth|D\.O\.B\.?)[\s:]*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4}|[0-9]{4}[-/][0-9]{2}[-/][0-9]{2})", re.IGNORECASE)
    GENDER_PATTERN = re.compile(r"\b(MALE|FEMALE|TRANSGENDER)\b", re.IGNORECASE)
    PINCODE_PATTERN = re.compile(r"\b([1-9][0-9]{5})\b")

    @classmethod
    def detect_document_type(cls, text: str) -> KYCDocumentType:
        text_upper = text.upper()
        if "UNIQUE IDENTIFICATION AUTHORITY OF INDIA" in text_upper or "AADHAAR" in text_upper or "VID :" in text_upper:
            return KYCDocumentType.AADHAAR_CARD
        if "INCOME TAX DEPARTMENT" in text_upper or "PERMANENT ACCOUNT NUMBER" in text_upper or "PAN" in text_upper:
            if cls.PAN_PATTERN.search(text_upper):
                return KYCDocumentType.PAN_CARD
        if "ELECTION COMMISSION OF INDIA" in text_upper or "ELECTOR" in text_upper or "VOTER" in text_upper:
            return KYCDocumentType.VOTER_ID
        if "DRIVING LICENCE" in text_upper or "UNION OF INDIA DRIVING LICENCE" in text_upper or "MOTOR VEHICLES" in text_upper:
            return KYCDocumentType.DRIVING_LICENCE
        if "REPUBLIC OF INDIA" in text_upper and "PASSPORT" in text_upper:
            return KYCDocumentType.PASSPORT

        # Fallback check on ID patterns
        if cls.AADHAAR_PATTERN.search(text):
            return KYCDocumentType.AADHAAR_CARD
        if cls.PAN_PATTERN.search(text):
            return KYCDocumentType.PAN_CARD
        if cls.VOTER_ID_PATTERN.search(text):
            return KYCDocumentType.VOTER_ID
        if cls.DL_PATTERN.search(text):
            return KYCDocumentType.DRIVING_LICENCE
        if cls.PASSPORT_PATTERN.search(text):
            return KYCDocumentType.PASSPORT

        return KYCDocumentType.UNKNOWN

    @classmethod
    def extract(cls, raw_text: str) -> ExtractedKYCData:
        doc_type = cls.detect_document_type(raw_text)
        doc_number = None
        full_name = None
        father_name = None
        dob = None
        gender = None
        pincode = None
        address_line = None
        city = None
        state = None

        # Extract Document Number based on type
        if doc_type == KYCDocumentType.AADHAAR_CARD:
            m = cls.AADHAAR_PATTERN.search(raw_text)
            if m:
                doc_number = DocumentNormalizer.normalize_aadhaar(m.group(1))
        elif doc_type == KYCDocumentType.PAN_CARD:
            m = cls.PAN_PATTERN.search(raw_text)
            if m:
                doc_number = DocumentNormalizer.normalize_pan(m.group(1))
        elif doc_type == KYCDocumentType.VOTER_ID:
            m = cls.VOTER_ID_PATTERN.search(raw_text)
            if m:
                doc_number = m.group(1).upper()
        elif doc_type == KYCDocumentType.DRIVING_LICENCE:
            m = cls.DL_PATTERN.search(raw_text)
            if m:
                doc_number = m.group(1).upper()
        elif doc_type == KYCDocumentType.PASSPORT:
            m = cls.PASSPORT_PATTERN.search(raw_text)
            if m:
                doc_number = m.group(1).upper()

        # DOB match
        m_dob = cls.DOB_PATTERN.search(raw_text)
        if m_dob:
            dob = DocumentNormalizer.normalize_date(m_dob.group(1))

        # Gender match
        m_gender = cls.GENDER_PATTERN.search(raw_text)
        if m_gender:
            gender = m_gender.group(1).upper()

        # Pincode match
        m_pin = cls.PINCODE_PATTERN.findall(raw_text)
        if m_pin:
            pincode = m_pin[-1]  # Typically at the end of address block

        # Name extraction heuristics from text lines
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            line_up = line.upper()
            if any(skip in line_up for skip in ["GOVERNMENT", "INDIA", "AUTHORITY", "DEPARTMENT", "INCOME TAX", "ELECTION", "PASSPORT", "LICENCE", "DOB", "MALE", "FEMALE", "VID :", "ADDRESS", "FATHER"]):
                continue
            # Check if looks like a name (2 to 4 capitalized words)
            words = line.split()
            if 1 <= len(words) <= 4 and re.match(r"^[A-Za-z\.\s]+$", line):
                full_name = line.strip().upper()
                # Next line might be Father's name if PAN
                if doc_type == KYCDocumentType.PAN_CARD and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not any(skip in next_line.upper() for skip in ["DOB", "INCOME", "PERMANENT"]) and re.match(r"^[A-Za-z\.\s]+$", next_line):
                        father_name = next_line.upper()
                break

        # Address extraction heuristic
        if "Address:" in raw_text or "ADDRESS" in raw_text.upper():
            addr_idx = raw_text.find("Address:")
            if addr_idx == -1:
                addr_idx = raw_text.upper().find("ADDRESS")
            addr_part = raw_text[addr_idx:].split("\n")
            addr_lines = [l.strip() for l in addr_part[1:4] if l.strip() and not cls.AADHAAR_PATTERN.search(l) and not "VID" in l]
            if addr_lines:
                address_line = ", ".join(addr_lines)

        addr_obj = None
        if address_line or pincode:
            addr_obj = Address(
                line1=address_line or "Address as per ID",
                pincode=pincode,
                city=city,
                state=state,
            )

        confidence = 0.95 if (doc_number and full_name) else (0.80 if doc_number else 0.50)

        return ExtractedKYCData(
            document_type=doc_type,
            document_number=doc_number,
            full_name=full_name,
            father_name=father_name,
            date_of_birth=dob,
            gender=gender,
            address=addr_obj,
            raw_text=raw_text,
            extraction_confidence=confidence,
        )


class SimulatedDatasetOCREngine(BaseOCREngine):
    """Engine that reads from the synthetic dataset or parses incoming text."""

    def extract_text(self, document_source: Any) -> str:
        if isinstance(document_source, str):
            return document_source
        if isinstance(document_source, dict):
            if "ocr_input" in document_source and "raw_ocr_text" in document_source["ocr_input"]:
                return document_source["ocr_input"]["raw_ocr_text"]
            if "raw_ocr_text" in document_source:
                return document_source["raw_ocr_text"]
        return str(document_source)

    def parse_kyc_fields(self, raw_text: str) -> ExtractedKYCData:
        return HeuristicRegexExtractor.extract(raw_text)
