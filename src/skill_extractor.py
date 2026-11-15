"""NLP skill extraction.

Method (fast enough for 100k+ long job descriptions):
  1. Normalise text (lower-case, fix "c++", "c#", ".net", "node.js", split on punctuation).
  2. Look up 1-to-N word phrases in an alias dictionary built from the taxonomy,
     longest match first, non-overlapping (so "react native" is not also "react").
  3. Return a sparse binary matrix: rows = postings, columns = skills.

Also: TF-IDF based discovery of frequent terms that are NOT in the taxonomy,
to help you decide which new skills to add (they are suggestions only).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RANDOM_STATE
from .skills_taxonomy import TAXONOMY

TOKEN_RE = re.compile(r"[a-z0-9]+")
# order matters: specific replacements first
_REPLACEMENTS = [("c++", "cplusplus"), ("c#", "csharp"), ("f#", "fsharp"),
                 (".net", " dotnet "), (".js", "js"), ("&", " and ")]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


class SkillExtractor:
    """Dictionary + n-gram skill extractor built from the taxonomy."""

    def __init__(self, taxonomy: Optional[dict] = None, extra_aliases: Optional[Dict[str, List[str]]] = None):
        taxonomy = taxonomy or TAXONOMY
        self.skill_names: List[str] = []
        self.categories: Dict[str, str] = {}
        self.alias_map: Dict[str, str] = {}          # "power bi" -> "Power BI"
        self.first_len: Dict[str, int] = {}          # first token -> longest alias starting with it
        for category, skills in taxonomy.items():
            for skill, aliases in skills.items():
                self.skill_names.append(skill)
                self.categories[skill] = category
                self._add(skill, aliases)
        for skill, aliases in (extra_aliases or {}).items():
            if skill in self.categories:
                self._add(skill, aliases)
        self.skill_index = {s: i for i, s in enumerate(self.skill_names)}

    def _add(self, skill: str, aliases: Iterable[str]) -> None:
        for alias in aliases:
            toks = tokenize(alias)
            if not toks:
                continue
            self.alias_map.setdefault(" ".join(toks), skill)
            self.first_len[toks[0]] = max(self.first_len.get(toks[0], 0), len(toks))

    # ------------------------------------------------------------------
    def extract_text(self, text: str) -> Set[str]:
        """Return the set of skills found in one text."""
        if not text:
            return set()
        tokens = TOKEN_RE.findall(normalize_text(text))
        n, i, found = len(tokens), 0, set()
        alias_map, first_len = self.alias_map, self.first_len
        while i < n:
            tok = tokens[i]
            max_len = first_len.get(tok)
            if max_len:
                matched = False
                for length in range(min(max_len, n - i), 0, -1):
                    key = tok if length == 1 else " ".join(tokens[i:i + length])
                    skill = alias_map.get(key)
                    if skill:
                        found.add(skill)
                        i += length
                        matched = True
                        break
                if matched:
                    continue
            i += 1
        return found

    def extract_many(self, texts: Iterable[str], n_total: Optional[int] = None,
                     progress: Optional[Callable[[float], None]] = None) -> csr_matrix:
        """Binary sparse matrix (n_texts x n_skills, dtype uint8)."""
        texts = list(texts) if n_total is None else texts
        n = n_total if n_total is not None else len(texts)
        indptr, indices = [0], []
        for i, text in enumerate(texts):
            cols = sorted(self.skill_index[s] for s in self.extract_text(text))
            indices.extend(cols)
            indptr.append(len(indices))
            if progress and i % 5000 == 0:
                progress(i / max(n, 1))
        data = np.ones(len(indices), dtype=np.uint8)
        return csr_matrix((data, indices, indptr), shape=(n, len(self.skill_names)))


# ----------------------------------------------------------------------
GENERIC_TERMS = set("""
experience work working team teams ability skills skill strong years year required preferred
including excellent role roles responsibilities requirements qualifications company benefits
position job candidate candidates opportunity employees employee business customer customers
manage management develop development support provide ensure insurance health equal employer
degree bachelor master education knowledge understanding communication written verbal
looking join environment training time full part new like need needs well best great
""".split())


def discover_candidate_terms(texts: pd.Series, extractor: SkillExtractor,
                             sample_n: int = 15_000, top_n: int = 100) -> pd.DataFrame:
    """TF-IDF discovery of frequent terms that are not already in the taxonomy.

    These are SUGGESTIONS for you to review manually; they are not counted as skills.
    """
    texts = texts[texts.str.len() > 0]
    if len(texts) < 50:
        return pd.DataFrame(columns=["term", "mean_tfidf", "postings_in_sample", "note"])
    sample = texts.sample(min(sample_n, len(texts)), random_state=RANDOM_STATE)
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=max(5, int(0.001 * len(sample))), max_df=0.15,
        stop_words="english", max_features=30_000, dtype=np.float32,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{2,}\b",
    )
    try:
        matrix = vec.fit_transform(sample)
    except ValueError:                       # empty vocabulary
        return pd.DataFrame(columns=["term", "mean_tfidf", "postings_in_sample", "note"])
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    known_tokens = {t for key in extractor.alias_map for t in key.split()}
    rows = []
    for term, m, d in zip(vec.get_feature_names_out(), mean, doc_freq):
        toks = TOKEN_RE.findall(term.lower())
        if any(t in known_tokens or t in GENERIC_TERMS for t in toks):
            continue
        rows.append((term, float(m), int(d)))
    out = pd.DataFrame(rows, columns=["term", "mean_tfidf", "postings_in_sample"])
    out = out.sort_values("mean_tfidf", ascending=False).head(top_n).reset_index(drop=True)
    out["note"] = "Candidate only - review manually before adding to the taxonomy"
    return out
