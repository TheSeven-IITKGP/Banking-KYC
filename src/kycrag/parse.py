"""Turn raw documents into paragraph records that keep the regulatory structure.

RBI Directions follow:  Chapter -> lettered Section (A., B., F.1) -> numbered
paragraph (1., 2., ...) -> clauses ((1), (i), (a)). We keep that hierarchy so every
chunk can be cited as "[doc, para N]".

The rules here were written against the RBI web page layout as of Nov 2026.
Always eyeball `data/processed/<doc_id>.paras.jsonl` after parsing; fix rules if
paragraphs are merged or split wrongly (see `python -m kycrag.cli inspect`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

INLINE_TAGS = ["a", "b", "strong", "i", "em", "u", "span", "font", "sup", "sub", "small", "big"]

CHAPTER_RE = re.compile(r"^chapter\s+([IVXL]+)\b\s*[–\-:.]?\s*(.*)$", re.I)
SECTION_RE = re.compile(r"^([A-Z])\.\s+(\S.{1,98})$|^([A-Z]\.\d+)\s+(\S.{1,98})$")
PARA_RE = re.compile(r"^(\d{1,3})\.\s+(\S.*)$")
ANNEX_RE = re.compile(r"^annex(ure)?\s*[–\-]?\s*([IVXL\d]+)\b", re.I)
FOOTNOTE_RE = re.compile(r"^\d{1,3}\s+(Inserted|Amended|Substituted|Added|Deleted|Updated|Modified)\b", re.I)
INTRO_RE = re.compile(r"^\d*\s*introduction\s*$", re.I)
TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$|\s{2,}\d{1,3}\s*$")
FAQ_Q_RE = re.compile(r"^Q\.?\s*(\d{1,3})[.)]\s*(.*)$", re.I)
BANK_HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.{1,150})$")

@dataclass
class ParsedPara:
    ref: str
    chapter: str = ""
    section: str = ""
    lines: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        if re.fullmatch(r"#F\d+", a["href"]) and a.get_text(strip=True).isdigit():
            a.decompose()
    for name in INLINE_TAGS:
        for t in soup.find_all(name):
            t.unwrap()
    soup.smooth()
    root = soup.body or soup
    text = root.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]

def _start_index(lines: list[str]) -> int:
    """Skip the site menu and table of contents."""
    for i, ln in enumerate(lines):
        if INTRO_RE.match(ln):
            return i
    chapter_one = [i for i, ln in enumerate(lines) if CHAPTER_RE.match(ln) and CHAPTER_RE.match(ln).group(1).upper() == "I"]
    if len(chapter_one) >= 2:
        return chapter_one[1]
    return chapter_one[0] if chapter_one else 0

def parse_direction_lines(lines: list[str]) -> list[ParsedPara]:
    lines = lines[_start_index(lines):]
    paras: list[ParsedPara] = []
    chapter = section = ""
    current: ParsedPara | None = None
    last_num = 0
    footnotes: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if FOOTNOTE_RE.match(ln):
            footnotes.append(ln)
            i += 1
            continue
        if INTRO_RE.match(ln):
            current = ParsedPara(ref="Introduction", chapter="Introduction")
            paras.append(current)
            i += 1
            continue
        m = CHAPTER_RE.match(ln)
        if m:
            title = m.group(2).strip()
            if not title and i + 1 < len(lines) and len(lines[i + 1]) < 80:
                title = lines[i + 1].strip().title()
                i += 1
            chapter = f"Chapter {m.group(1).upper()} – {title}".strip(" –")
            section, current = "", None
            i += 1
            continue
        m = ANNEX_RE.match(ln)
        if m and len(ln) < 120:
            chapter, section = f"Annex {m.group(2)}", ""
            current = None
            i += 1
            continue
        m = SECTION_RE.match(ln)
        if m and not PARA_RE.match(ln) and len((m.group(2) or m.group(4)).split()) <= 10:
            label = m.group(1) or m.group(3)
            title = m.group(2) or m.group(4)
            section = (f"{label} {title}" if "." in label else f"{label}. {title}").rstrip(".:")
            current = None
            i += 1
            continue
        m = PARA_RE.match(ln)
        if m:
            n = int(m.group(1))
            if last_num < n <= last_num + 3:
                current = ParsedPara(ref=str(n), chapter=chapter, section=section, lines=[ln])
                paras.append(current)
                last_num = n
                i += 1
                continue
        if current is not None:
            current.lines.append(ln)
        i += 1
    if footnotes and paras:
        paras[-1].footnotes = footnotes
    return [p for p in paras if p.text]

def parse_faq_lines(lines: list[str]) -> list[ParsedPara]:
    paras: list[ParsedPara] = []
    current: ParsedPara | None = None
    for ln in lines:
        m = FAQ_Q_RE.match(ln)
        if m:
            current = ParsedPara(ref=f"FAQ {m.group(1)}", chapter="FAQs", lines=[ln])
            paras.append(current)
        elif current is not None:
            current.lines.append(ln)
    return paras

def pdf_to_lines(path) -> list[str]:
    import fitz

    doc = fitz.open(path)
    pages = [[re.sub(r"\s+", " ", ln).strip() for ln in page.get_text().splitlines()] for page in doc]
    pages = [[ln for ln in p if ln] for p in pages]
    counts: dict[str, int] = {}
    for p in pages:
        for ln in set(p):
            counts[ln] = counts.get(ln, 0) + 1
    repeated = {ln for ln, c in counts.items() if len(pages) > 3 and c > len(pages) / 2}
    out: list[str] = []
    for p in pages:
        for ln in p:
            if ln in repeated or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", ln, re.I):
                continue
            out.append(ln)
    return out

def parse_bank_lines(lines: list[str]) -> list[ParsedPara]:
    """Bank policies use dotted numbering (3, 3.1, 3.1.2). Each numbered heading
    starts a new record. Needs manual review: every bank formats differently."""
    paras: list[ParsedPara] = []
    current: ParsedPara | None = None
    chapter = ""
    for ln in lines:
        if TOC_LINE_RE.search(ln):
            continue
        m = CHAPTER_RE.match(ln)
        if m:
            chapter = f"Chapter {m.group(1).upper()} – {m.group(2).strip()}".strip(" –")
            continue
        m = BANK_HEAD_RE.match(ln)
        if m and not ln.endswith(";"):
            current = ParsedPara(ref=m.group(1), chapter=chapter, lines=[ln])
            paras.append(current)
        elif current is not None:
            current.lines.append(ln)
    seen, unique = set(), []
    for p in paras:
        if p.ref not in seen and len(p.text) > 40:
            seen.add(p.ref)
            unique.append(p)
    return unique

def parse_file(path, parser: str) -> list[ParsedPara]:
    if parser in ("rbi_html", "faq_html"):
        from pathlib import Path
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = html_to_lines(html)
        return parse_direction_lines(lines) if parser == "rbi_html" else parse_faq_lines(lines)
    if parser == "rbi_pdf":
        return parse_direction_lines(pdf_to_lines(path))
    if parser == "faq_pdf":
        return parse_faq_lines(pdf_to_lines(path))
    if parser == "bank_pdf":
        return parse_bank_lines(pdf_to_lines(path))
    raise ValueError(f"unknown parser {parser!r}")
