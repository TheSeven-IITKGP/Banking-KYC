"""Document Normalization and Field Sanitization for Indian KYC.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

import re
from typing import Optional


class DocumentNormalizer:
    """Standardizes dates, document numbers, and names across diverse Indian ID formats."""

    @staticmethod
    def normalize_name(name: Optional[str]) -> str:
        if not name:
            return ""
        # Remove extra spaces, honorifics (MR., MRS., MS., SHRI, SMT.)
        cleaned = re.sub(r"\b(MR|MRS|MS|SHRI|SMT|DR)\.?\b", "", name, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z\s\.]", "", cleaned)
        return " ".join(cleaned.split()).upper()

    @staticmethod
    def normalize_date(date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        date_str = date_str.strip()
        # Matches DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        m = re.match(r"^(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})$", date_str)
        if m:
            d, mth, y = m.groups()
            return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
        # Matches YYYY-MM-DD, YYYY/MM/DD
        m2 = re.match(r"^(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})$", date_str)
        if m2:
            y, mth, d = m2.groups()
            return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
        return date_str

    @staticmethod
    def normalize_aadhaar(aadhaar_str: Optional[str]) -> Optional[str]:
        if not aadhaar_str:
            return None
        cleaned = re.sub(r"\s+", "-", aadhaar_str.strip())
        return cleaned.upper()

    @staticmethod
    def normalize_pan(pan_str: Optional[str]) -> Optional[str]:
        if not pan_str:
            return None
        return pan_str.strip().upper()

    @staticmethod
    def calculate_levenshtein_ratio(s1: str, s2: str) -> float:
        """Calculates normalized Levenshtein similarity ratio between 0.0 and 1.0."""
        s1 = s1.strip().upper()
        s2 = s2.strip().upper()
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return max(0.0, 1.0 - (distance / max_len))
