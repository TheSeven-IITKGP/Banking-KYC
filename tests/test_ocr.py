"""Tests for OCR Preprocessing, Normalization, and Extraction.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

import unittest
from banking_kyc.models import KYCDocumentType
from banking_kyc.ocr.engine import HeuristicRegexExtractor
from banking_kyc.ocr.normalizer import DocumentNormalizer
from banking_kyc.ocr.preprocessor import ImageQualityAnalyzer, OCRNoiseSimulator

class TestOCRModule(unittest.TestCase):

    def test_aadhaar_extraction(self):
        text = (
            "GOVERNMENT OF INDIA\n"
            "Unique Identification Authority of India\n"
            "AARAV SHETTY\n"
            "DOB: 27-12-1975\n"
            "MALE\n"
            "Address:\n"
            "Flat 855, Tower A, Koramangala 4th Block,\n"
            "New Delhi, Delhi - 110027\n"
            "XXXX-XXXX-4811\n"
            "VID : 9279-1434-4257-9928"
        )
        res = HeuristicRegexExtractor.extract(text)
        self.assertEqual(res.document_type, KYCDocumentType.AADHAAR_CARD)
        self.assertEqual(res.document_number, "XXXX-XXXX-4811")
        self.assertEqual(res.full_name, "AARAV SHETTY")
        self.assertEqual(res.date_of_birth, "1975-12-27")
        self.assertEqual(res.gender, "MALE")
        self.assertIsNotNone(res.address)
        self.assertEqual(res.address.pincode, "110027")

    def test_pan_extraction(self):
        text = (
            "INCOME TAX DEPARTMENT\n"
            "GOVT. OF INDIA\n"
            "Permanent Account Number Card\n"
            "ABCDE1234F\n"
            "VIKRAM MALHOTRA\n"
            "RAJESH MALHOTRA\n"
            "DOB: 15/08/1988"
        )
        res = HeuristicRegexExtractor.extract(text)
        self.assertEqual(res.document_type, KYCDocumentType.PAN_CARD)
        self.assertEqual(res.document_number, "ABCDE1234F")
        self.assertEqual(res.date_of_birth, "1988-08-15")

    def test_image_quality_check(self):
        analyzer = ImageQualityAnalyzer(min_confidence_threshold=0.70)
        ok, flags, msg = analyzer.evaluate_quality("Standard length text over 20 chars long for checking", 0.92, False, False)
        self.assertTrue(ok)

        bad_ok, bad_flags, bad_msg = analyzer.evaluate_quality("Text long enough but blurred and poor score", 0.55, blur_detected=True)
        self.assertFalse(bad_ok)
        self.assertIn("Severe blur detected", bad_msg)

    def test_name_levenshtein(self):
        sim1 = DocumentNormalizer.calculate_levenshtein_ratio("Chandan Bauri", "Chandan Bauri")
        self.assertEqual(sim1, 1.0)
        sim2 = DocumentNormalizer.calculate_levenshtein_ratio("Aarav Shetty", "A. Shetty")
        self.assertGreater(sim2, 0.50)

if __name__ == "__main__":
    unittest.main()
