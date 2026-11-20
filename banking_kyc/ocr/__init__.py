"""OCR Processing Module for Indian Banking KYC Documents.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from banking_kyc.ocr.engine import BaseOCREngine, HeuristicRegexExtractor, SimulatedDatasetOCREngine
from banking_kyc.ocr.normalizer import DocumentNormalizer
from banking_kyc.ocr.preprocessor import ImageQualityAnalyzer, OCRNoiseSimulator

__all__ = [
    "BaseOCREngine",
    "SimulatedDatasetOCREngine",
    "HeuristicRegexExtractor",
    "DocumentNormalizer",
    "ImageQualityAnalyzer",
    "OCRNoiseSimulator",
]
