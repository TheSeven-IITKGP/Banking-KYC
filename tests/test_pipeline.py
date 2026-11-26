"""Offline tests: python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from kycrag.bm25 import BM25, tokenize
from kycrag.chunking import extract_glossary, extract_xrefs, naive_chunks, paragraph_chunks
from kycrag.config import SETTINGS
from kycrag.embeddings import HashEmbedder
from kycrag.generate import check_citations
from kycrag.guard import redact, understand
from kycrag.models import load_manifest
from kycrag.parse import html_to_lines, parse_direction_lines
from kycrag.retrieve import ABLATIONS, Hit, RetrievalConfig, Retriever, rrf

def parse_fixture(name="sample_direction.html"):
    return parse_direction_lines(html_to_lines((FIX / name).read_text(encoding="utf-8")))

class ParseTests(unittest.TestCase):
    def test_numbered_paragraphs_and_structure(self):
        paras = parse_fixture()
        refs = [p.ref for p in paras]
        self.assertEqual(refs, ["Introduction", "1", "2", "3", "4", "5", "6", "7", "8"])
        p5 = next(p for p in paras if p.ref == "5")
        self.assertEqual(p5.chapter, "Chapter II – Accounts")
        self.assertEqual(p5.section, "A. Basic accounts")
        self.assertIn("(2) the balance", p5.text)
        p8 = next(p for p in paras if p.ref == "8")
        self.assertEqual(p8.section, "F.1 Enhanced checks")

    def test_toc_and_menu_skipped_and_footnote_marker_removed(self):
        paras = parse_fixture()
        everything = " ".join(p.text for p in paras)
        self.assertNotIn("About Us", everything)
        p4 = next(p for p in paras if p.ref == "4")
        self.assertTrue(p4.text.endswith("paragraphs 6 and 7."))
        self.assertEqual(paras[-1].footnotes[0][:10], "1 Inserted")

    def test_inline_bold_does_not_split_lines(self):
        p1 = next(p for p in parse_fixture() if p.ref == "1")
        self.assertIn("Sample Directions, 2025", p1.text)

    def test_numbered_list_inside_paragraph_is_not_a_paragraph(self):
        p7 = next(p for p in parse_fixture() if p.ref == "7")
        self.assertIn("must not become a paragraph", p7.text)

    def test_2016_style_chapter_title_on_next_line(self):
        paras = parse_fixture("old_direction.html")
        self.assertEqual([p.ref for p in paras], ["Introduction", "1", "2"])
        self.assertEqual(paras[2].chapter, "Chapter II – Accounts")

class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_manifest(FIX / "manifest.csv")[0]
        self.paras = parse_fixture()

    def test_xrefs_extracted_and_resolved(self):
        self.assertEqual(extract_xrefs("limits in paragraph 5 and checks in paragraphs 6 and 7", "4"), ["5", "6", "7"])
        self.assertEqual(extract_xrefs("as per clause (5) of paragraph 42", "1"), ["42"])
        chunks = {c.chunk_id: c for c in paragraph_chunks(self.paras, self.doc, 2500)}
        self.assertEqual(chunks["NEW:p4"].cross_refs, ["NEW:p5", "NEW:p6", "NEW:p7"])

    def test_glossary(self):
        g = extract_glossary("(i) ‘Officially Valid Document (OVD)’ means x. (ii) ‘Central Test Registry’ (CTR) means y.")
        self.assertEqual(g, {"OVD": "Officially Valid Document", "CTR": "Central Test Registry"})

    def test_long_paragraph_split_at_clauses_with_lead(self):
        chunks = [c for c in paragraph_chunks(self.paras, self.doc, 400) if c.para_ref == "8"]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[1].text.startswith("(continued) 8. The bank shall apply"))
        self.assertEqual(chunks[0].chunk_id, "NEW:p8.0")

    def test_heading_path_and_definition_flag(self):
        c3 = next(c for c in paragraph_chunks(self.paras, self.doc, 2500) if c.para_ref == "3")
        self.assertTrue(c3.is_definition)
        self.assertIn("Chapter I – Preliminary > C. Definitions > Para 3", c3.heading_path)

    def test_naive_chunks_track_paragraphs(self):
        ch = naive_chunks(self.paras, self.doc, 300)
        self.assertIn("5", ",".join(c.para_ref for c in ch))

class GuardTests(unittest.TestCase):
    def test_redaction(self):
        r = redact("PAN ABCDE1234F, Aadhaar 2345 6789 0123, phone 9876543210, mail a.b@x.com")
        self.assertEqual(set(r.found), {"PAN", "AADHAAR", "PHONE", "EMAIL"})
        for raw in ("ABCDE1234F", "2345 6789 0123", "9876543210", "a.b@x.com"):
            self.assertNotIn(raw, r.text)

    def test_amounts_are_not_pii(self):
        self.assertEqual(redact("limit of ₹50,000 and ₹1,00,000 in 2026").found, [])

    def test_intents(self):
        self.assertEqual(understand("What is the small account limit?", {}).intent, "in_scope")
        self.assertEqual(understand("Check the KYC status of my account", {}).intent, "customer_specific")
        self.assertEqual(understand("Write a poem about the monsoon", {}).intent, "out_of_scope")
        self.assertTrue(understand("How did the 2016 direction differ?", {}).include_superseded)
        q = understand("What counts as an OVD?", {"OVD": "Officially Valid Document"})
        self.assertIn("Officially Valid Document", q.expanded)

class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.saved = (SETTINGS.raw_dir, SETTINGS.processed_dir, SETTINGS.manifest_path, SETTINGS.log_path)
        SETTINGS.raw_dir, SETTINGS.processed_dir = FIX, cls.tmp / "processed"
        SETTINGS.manifest_path, SETTINGS.log_path = FIX / "manifest.csv", cls.tmp / "q.jsonl"
        from kycrag.indexer import build_all
        cls.meta = build_all("hash")
        cls.r = Retriever("paragraph", embedder=HashEmbedder(), reranker=FakeReranker())

    @classmethod
    def tearDownClass(cls):
        SETTINGS.raw_dir, SETTINGS.processed_dir, SETTINGS.manifest_path, SETTINGS.log_path = cls.saved
        shutil.rmtree(cls.tmp)

    def test_index_built(self):
        self.assertEqual(self.meta["docs"]["NEW"]["paragraphs"], 9)
        self.assertIn("CTR", json.loads((SETTINGS.processed_dir / "glossary.json").read_text()))

    def test_version_filter(self):
        full = ABLATIONS[-1]
        ids = [h.chunk.chunk_id for h in self.r.search("basic account balance limit", full)]
        self.assertTrue(all(not i.startswith("OLD:") for i in ids))
        ids_hist = [h.chunk.chunk_id for h in self.r.search("basic account balance limit", full, include_superseded=True)]
        self.assertTrue(any(i.startswith("OLD:") for i in ids_hist))
        no_filter = RetrievalConfig("x", rerank=False, xrefs=False, version_filter=False)
        self.assertTrue(any(h.chunk.doc_id == "OLD" for h in self.r.search("basic account balance limit", no_filter)))

    def test_xref_expansion_adds_linked(self):
        cfg = RetrievalConfig("x", rerank=False, xrefs=True)
        hits = self.r.search("open a basic account identity checks", cfg)
        linked = [h.chunk.chunk_id for h in hits if h.via == "linked"]
        self.assertTrue(set(linked) <= {"NEW:p5", "NEW:p6", "NEW:p7", "NEW:p3"})

    def test_bm25_and_rrf(self):
        bm = BM25(["small account limit", "video verification"])
        self.assertEqual(bm.top_k("video", 1)[0][0], 1)
        self.assertEqual(tokenize("₹50,000 limit"), ["rs", "50000", "limit"])
        self.assertEqual(rrf([[(1, 0.9), (2, 0.5)], [(2, 3.0), (3, 1.0)]])[0][0], 2)

    def test_pipeline_good_and_bad_citations(self):
        from kycrag.pipeline import KYCAssistant
        good = KYCAssistant(self.r, FakeLLM(cite_first=True), "test").ask("What is the basic account balance limit?")
        self.assertEqual(good["status"], "found")
        self.assertTrue(good["citations"])
        bad = KYCAssistant(self.r, FakeLLM(cite_first=False), "test").ask("What is the basic account balance limit?")
        self.assertEqual(bad["status"], "unverified")
        refused = KYCAssistant(self.r, FakeLLM(True), "test").ask("Write a poem about the monsoon")
        self.assertEqual(refused["status"], "refused")
        log = SETTINGS.log_path.read_text()
        self.assertNotIn("Write a poem", log.split("\n")[0])

    def test_pii_never_logged(self):
        from kycrag.pipeline import KYCAssistant
        KYCAssistant(self.r, FakeLLM(True), "test").ask("Customer PAN ABCDE1234F: which account limits apply?")
        self.assertNotIn("ABCDE1234F", SETTINGS.log_path.read_text())

    def test_check_citations(self):
        c = self.r.chunks[0]
        hits = [Hit(c, 1.0)]
        self.assertEqual(check_citations({"answer": f"x [{c.chunk_id}]", "citations": []}, hits), [])
        self.assertEqual(check_citations({"answer": "x [NEW:p99]", "citations": []}, hits), ["NEW:p99"])

    def test_eval_recall(self):
        from run_eval import recall
        c5 = self.r.by_id["NEW:p5"]
        self.assertEqual(recall({("NEW", "5"), ("NEW", "6")}, [c5]), 0.5)

class FakeReranker:
    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]

class FakeLLM:
    model = "fake"

    def __init__(self, cite_first: bool):
        self.cite_first = cite_first

    def complete(self, system, user):
        first = user.split("[", 1)[1].split("]", 1)[0]
        tag = first if self.cite_first else "NEW:p999"
        return json.dumps({"status": "found", "answer": f"The limit is stated here [{tag}].", "citations": [tag], "conflicts": ""})

if __name__ == "__main__":
    unittest.main()
