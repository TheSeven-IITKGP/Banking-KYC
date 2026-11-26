"""Input guard (PII redaction) and lightweight query understanding.

Everything here is rule-based: zero API cost and easy to explain in the report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .timing import timer

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[\s-]){3}\d{1,7}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"(?<![\d,])\d{9,18}(?![\d,])")),
]

CUSTOMER_SPECIFIC = re.compile(
    r"\b(my|his|her|their)\s+(account(?!\s*number)|kyc|application|status|documents?|loan|pan\s*card|identity|profile)\b"
    r"|\b(my|his|her|their)\s+\w+'s\s+(account|kyc|application|status|documents?|loan|identity|profile)\b"
    r"|\b(this\s+customer'?s?|customer\s+named|client\s+named|customer\s+\w+(?:\s+\w+)?'s)\b"
    r"|\b(pull up|retrieve|fetch|look up)\s+(the\s+)?(kyc|account|customer)\b"
    r"|\bstatus of (the )?(kyc|account) (of|for)\b|\bcheck (the )?kyc (of|for)\b",
    re.I,
)
HISTORY_INTENT = re.compile(
    r"\b(2016|old(er)?|earlier|previous(ly)?|superseded|repealed|change[sd]?|difference|differ|before|"
    r"amend(ed|ment)?|history|used to)\b",
    re.I,
)
DOMAIN_WORDS = set(
    "kyc aml cft cdd edd due diligence customer account accounts bank banks document documents ovd aadhaar pan "
    "identity address proof beneficial owner pep politically exposed risk updation periodic ckycr v-cip vcip video "
    "verification verify identification onboarding legal entity company trust partnership proprietor proprietary "
    "nri pio foreign student minor shg fpi small str ctr suspicious transaction transactions record records fiu "
    "sanctions uapa wmd wire transfer correspondent rbi direction directions paragraph para regulated entity "
    "digital otp e-kyc ekyc offline photograph signature form 60 money laundering terrorist financing "
    "cheque cheques draft drafts demand payee payees purchaser instrument instruments".split()
)
CUSTOMER_TYPES = {
    "individual": r"\bindividuals?\b", "sole proprietary firm": r"\bproprietor(ship)?|proprietary\b",
    "company": r"\bcompan(y|ies)\b", "partnership firm": r"\bpartnership\b", "trust": r"\btrusts?\b",
    "NRI / PIO": r"\b(nri|pio|non-resident)\b", "foreign student": r"\bforeign students?\b",
    "PEP": r"\b(pep|politically exposed)\b", "SHG": r"\b(shg|self help group)\b", "FPI": r"\b(fpi|foreign portfolio)\b",
}

@dataclass
class GuardResult:
    text: str
    found: list[str] = field(default_factory=list)

    @property
    def had_pii(self) -> bool:
        return bool(self.found)

def redact(text: str) -> GuardResult:
    found: list[str] = []
    for label, rx in PII_PATTERNS:
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return GuardResult(text=text, found=found)

@dataclass
class QueryInfo:
    original: str
    redacted: str
    pii_found: list[str]
    intent: str
    include_superseded: bool
    customer_types: list[str]
    expanded: str

def understand(query: str, glossary: dict[str, str]) -> QueryInfo:
    with timer("understand"):
        g = redact(query)
        q = g.text
        words = set(re.findall(r"[a-z0-9\-]+", q.lower()))
        if CUSTOMER_SPECIFIC.search(q):
            intent = "customer_specific"
        elif words & DOMAIN_WORDS or any(a.lower() in words for a in glossary):
            intent = "in_scope"
        else:
            intent = "out_of_scope"
        expansions = [f"{a} ({t})" for a, t in glossary.items() if re.search(rf"\b{re.escape(a)}\b", q)]
        ctypes = [name for name, rx in CUSTOMER_TYPES.items() if re.search(rx, q, re.I)]
        expanded = q + ("\n" + "; ".join(expansions) if expansions else "")
        return QueryInfo(query, q, g.found, intent, bool(HISTORY_INTENT.search(q)), ctypes, expanded)
