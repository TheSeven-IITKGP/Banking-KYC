"""Okapi BM25 in ~50 lines, so the project has no extra dependency."""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set(
    "a an the of to in for on by and or is are be shall with as at from that this these those it its "
    "which who whom such any all not no if than then there their them they has have had was were".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("₹", " rs ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 50,000 -> 50000
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if t.isdigit() or (t not in STOP and len(t) > 1)]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.len) / max(len(self.len), 1)
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top_k(self, query: str, k: int, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        sc = self.scores(query)
        idx = [i for i in range(len(sc)) if (allowed is None or allowed[i]) and sc[i] > 0]
        idx.sort(key=lambda i: sc[i], reverse=True)
        return [(i, sc[i]) for i in idx[:k]]
