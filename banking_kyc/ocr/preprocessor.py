"""OCR Preprocessing and Quality Analysis.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import random
from typing import Dict, Tuple
from banking_kyc.models import ImageQualityFlags

class ImageQualityAnalyzer:
    """Analyzes image quality indicators for KYC document OCR."""

    def __init__(self, min_confidence_threshold: float = 0.70):
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate_quality(
        self,
        raw_text: str,
        simulated_confidence: float = 0.95,
        blur_detected: bool = False,
        glare_detected: bool = False,
        orientation_angle: int = 0,
    ) -> Tuple[bool, ImageQualityFlags, str]:
        """Evaluates whether the document image quality is sufficient for KYC processing."""
        text_length = len(raw_text.strip()) if raw_text else 0
        if text_length < 20:
            return (
                False,
                ImageQualityFlags(
                    blur_detected=True,
                    glare_detected=glare_detected,
                    orientation_angle=orientation_angle,
                    simulated_ocr_confidence=min(simulated_confidence, 0.40),
                ),
                "Insufficient text extracted from document image (too sparse or unreadable)",
            )

        if (blur_detected or glare_detected) and simulated_confidence < self.min_confidence_threshold:
            reasons = []
            if blur_detected:
                reasons.append("Severe blur detected")
            if glare_detected:
                reasons.append("Severe glare detected")
            reasons.append(f"Confidence {simulated_confidence:.2f} is below threshold {self.min_confidence_threshold:.2f}")
            return (
                False,
                ImageQualityFlags(
                    blur_detected=blur_detected,
                    glare_detected=glare_detected,
                    orientation_angle=orientation_angle,
                    simulated_ocr_confidence=simulated_confidence,
                ),
                "; ".join(reasons),
            )

        return (
            True,
            ImageQualityFlags(
                blur_detected=blur_detected,
                glare_detected=glare_detected,
                orientation_angle=orientation_angle,
                simulated_ocr_confidence=simulated_confidence,
            ),
            "Quality acceptable",
        )

class OCRNoiseSimulator:
    """Simulates realistic optical character recognition errors found in Indian ID scans."""

    OCR_CHAR_MAP: Dict[str, list[str]] = {
        "O": ["0", "Q"],
        "0": ["O", "D"],
        "I": ["1", "l", "|"],
        "1": ["I", "l"],
        "S": ["5", "$"],
        "5": ["S"],
        "8": ["B"],
        "B": ["8"],
        "Z": ["2"],
        "2": ["Z"],
    }

    @classmethod
    def inject_noise(cls, text: str, error_rate: float = 0.05, seed: int | None = None) -> str:
        """Injects realistic OCR confusion noise into text."""
        if seed is not None:
            random.seed(seed)
        chars = list(text)
        for i, char in enumerate(chars):
            upper_char = char.upper()
            if upper_char in cls.OCR_CHAR_MAP and random.random() < error_rate:
                replacement = random.choice(cls.OCR_CHAR_MAP[upper_char])
                chars[i] = replacement.lower() if char.islower() else replacement
        return "".join(chars)
