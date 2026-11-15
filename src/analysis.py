"""Descriptive skill analytics computed from the sparse skill matrix.

All functions take plain numpy/pandas/scipy objects and return DataFrames.
Counts are real counts from your data; nothing here is simulated.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu


def skill_demand(X: csr_matrix, skill_names: List[str], categories: Dict[str, str],
                 min_postings: int = 1) -> pd.DataFrame:
    """Postings per skill and share of all postings (0-1)."""
    n_docs = X.shape[0]
    counts = np.asarray(X.sum(axis=0)).ravel().astype(int)
    df = pd.DataFrame({"skill": skill_names,
                       "category": [categories.get(s, "Other") for s in skill_names],
                       "postings": counts})
    df["share"] = df["postings"] / max(n_docs, 1)
    df = df[df["postings"] >= min_postings].sort_values("postings", ascending=False)
    return df.reset_index(drop=True)


def category_demand(X: csr_matrix, skill_names: List[str], categories: Dict[str, str]) -> pd.DataFrame:
    """Postings mentioning at least one skill of each category."""
    n_docs = X.shape[0]
    rows = []
    for cat in sorted(set(categories.values())):
        idx = [i for i, s in enumerate(skill_names) if categories.get(s) == cat]
        if not idx:
            continue
        has_any = np.asarray(X[:, idx].sum(axis=1)).ravel() > 0
        rows.append((cat, len(idx), int(has_any.sum()), has_any.sum() / max(n_docs, 1)))
    return pd.DataFrame(rows, columns=["category", "skills_in_category", "postings", "share"]) \
        .sort_values("postings", ascending=False).reset_index(drop=True)


def cooccurrence(X: csr_matrix, skill_names: List[str], top_n: int = 40) -> pd.DataFrame:
    """Co-occurrence COUNT matrix of the top_n most frequent skills (diagonal = skill count)."""
    counts = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(-counts)[:top_n]
    idx = idx[counts[idx] > 0]
    sub = X[:, idx].astype(np.int32)             # int32 avoids uint8 overflow
    mat = (sub.T @ sub).toarray()
    names = [skill_names[i] for i in idx]
    return pd.DataFrame(mat, index=names, columns=names)


def role_skill_profile(X: csr_matrix, roles: pd.Series, skill_names: List[str],
                       min_role_postings: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Share (0-1) of each role's postings that mention each skill."""
    counts = roles[roles != "Other"].value_counts()
    counts = counts[counts >= min_role_postings]
    rows, kept = {}, {}
    role_arr = roles.to_numpy()
    for role, n in counts.items():
        mask = role_arr == role
        rows[role] = np.asarray(X[mask].mean(axis=0)).ravel()
        kept[role] = int(n)
    profile = pd.DataFrame.from_dict(rows, orient="index", columns=skill_names)
    return profile, pd.Series(kept, name="postings")


def seniority_skew(X: csr_matrix, level_group: pd.Series, skill_names: List[str],
                   min_group: int, min_count: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
    """Compare skill share in junior vs senior postings (log-ratio, smoothed).

    Returns (dataframe or None, reason). log_ratio > 0 -> more common in junior postings.
    """
    lg = level_group.to_numpy()
    jm, sm = lg == "junior", lg == "senior"
    nj, ns = int(jm.sum()), int(sm.sum())
    if nj < min_group or ns < min_group:
        return None, (f"Not enough labelled postings (junior={nj}, senior={ns}; "
                      f"need >= {min_group} each).")
    jc = np.asarray(X[jm].sum(axis=0)).ravel()
    sc = np.asarray(X[sm].sum(axis=0)).ravel()
    df = pd.DataFrame({"skill": skill_names, "junior_n": jc.astype(int), "senior_n": sc.astype(int)})
    df["junior_share"] = df["junior_n"] / nj
    df["senior_share"] = df["senior_n"] / ns
    df["log_ratio"] = np.log(((df["junior_n"] + 0.5) / (nj + 1)) / ((df["senior_n"] + 0.5) / (ns + 1)))
    df = df[(df["junior_n"] + df["senior_n"]) >= min_count].copy()
    df["lean"] = np.where(df["log_ratio"] > 0.3, "junior-leaning",
                          np.where(df["log_ratio"] < -0.3, "senior-leaning", "balanced"))
    return df.sort_values("log_ratio", ascending=False).reset_index(drop=True), \
        f"junior postings={nj}, senior postings={ns}"


def salary_premium(X: csr_matrix, salary: pd.Series, skill_names: List[str],
                   min_n: int = 40) -> Tuple[Optional[pd.DataFrame], str]:
    """Median salary with vs without each skill (association only, NOT causation)."""
    sal = salary.to_numpy(dtype=float)
    mask = ~np.isnan(sal)
    if mask.sum() < 2 * min_n:
        return None, f"Only {int(mask.sum())} postings have a usable salary."
    Xs = X[mask].tocsc()
    y = sal[mask]
    rows = []
    for j, name in enumerate(skill_names):
        has = np.zeros(len(y), dtype=bool)
        has[Xs.indices[Xs.indptr[j]:Xs.indptr[j + 1]]] = True
        n_with, n_without = int(has.sum()), int((~has).sum())
        if n_with < min_n or n_without < min_n:
            continue
        med_with, med_without = float(np.median(y[has])), float(np.median(y[~has]))
        try:
            p = float(mannwhitneyu(y[has], y[~has], alternative="two-sided").pvalue)
        except ValueError:
            p = np.nan
        rows.append((name, n_with, med_with, med_without,
                     (med_with / med_without - 1) * 100 if med_without else np.nan, p))
    if not rows:
        return None, "No skill had enough salary-labelled postings."
    out = pd.DataFrame(rows, columns=["skill", "postings_with_salary", "median_with", "median_without",
                                      "premium_pct", "p_value"])
    return out.sort_values("premium_pct", ascending=False).reset_index(drop=True), \
        f"{int(mask.sum())} postings with salary"
