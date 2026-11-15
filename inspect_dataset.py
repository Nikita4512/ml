"""Print a quick, honest report about your CSV BEFORE running the full pipeline.

    python inspect_dataset.py                      # auto-finds the CSV in data/raw/
    python inspect_dataset.py path/to/postings.csv
"""
import sys
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.data_loader import (DatasetError, detect_schema, find_postings_file, parse_datetime,
                             read_header)
from src.trends import assess_temporal


def main() -> int:
    try:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_postings_file(RAW_DIR)
        print(f"File: {path}  ({path.stat().st_size / 1e6:.1f} MB)\n")
        cols = read_header(path)
        mapping = detect_schema(cols)
        print(f"{len(cols)} columns found:\n  " + ", ".join(cols))
        print("\nAUTOMATIC COLUMN DETECTION (canonical -> your column):")
        for k in ("title", "description", "skills_desc", "listed_time", "experience_level", "work_type",
                  "normalized_salary", "min_salary", "max_salary", "pay_period", "company", "location", "job_id"):
            print(f"  {k:18} -> {mapping.get(k, '-- NOT FOUND --')}")
        if not any(k in mapping for k in ("title", "description", "skills_desc")):
            print("\nPROBLEM: no text column detected. Skill extraction needs a description/title column.")
            return 1

        # read only the columns needed for a fast full-file scan (no descriptions)
        scan_cols = [mapping[k] for k in ("listed_time", "experience_level", "normalized_salary",
                                          "min_salary", "max_salary", "med_salary") if k in mapping]
        rows, nulls, dates, levels, sal_ok = 0, pd.Series(dtype=float), [], pd.Series(dtype=int), 0
        if scan_cols:
            for chunk in pd.read_csv(path, usecols=scan_cols, dtype=str, chunksize=100_000,
                                     encoding="utf-8", encoding_errors="replace", on_bad_lines="skip"):
                rows += len(chunk)
                if "listed_time" in mapping:
                    dates.append(parse_datetime(chunk[mapping["listed_time"]]))
                if "experience_level" in mapping:
                    levels = levels.add(chunk[mapping["experience_level"]].fillna("(missing)").value_counts(), fill_value=0)
                sal_cols = [mapping[k] for k in ("normalized_salary", "min_salary", "max_salary", "med_salary") if k in mapping]
                if sal_cols:
                    sal_ok += int(chunk[sal_cols].notna().any(axis=1).sum())
        else:
            rows = sum(len(c) for c in pd.read_csv(path, usecols=[cols[0]], chunksize=200_000))
        print(f"\nTotal rows: {rows:,}")
        if sal_ok:
            print(f"Rows with any salary info: {sal_ok:,} ({sal_ok / rows * 100:.1f}%)")
        if len(levels):
            print("\nExperience levels:\n" + levels.astype(int).sort_values(ascending=False).to_string())
        if dates:
            all_dates = pd.concat(dates)
            valid = all_dates.dropna()
            print(f"\nValid dates: {len(valid):,} ({len(valid) / rows * 100:.1f}%)")
            if len(valid):
                print(f"Date range : {valid.min()} -> {valid.max()}")
                print("Postings per month:\n" + valid.dt.to_period('M').value_counts().sort_index().to_string())
            t = assess_temporal(all_dates)
            print(f"\nTEMPORAL VERDICT: Mode {t['mode']}")
            for r in t["reasons"]:
                print("  - " + r)
        else:
            print("\nNo date column found: time trends and forecasts will be disabled (Mode B).")

        helpers = [p for p in path.parent.rglob("*.csv") if p != path]
        if helpers:
            print("\nOther CSV files nearby (NOT used as skills by this project):")
            for p in helpers[:15]:
                try:
                    h = list(pd.read_csv(p, nrows=3).columns)
                except Exception:
                    h = ["(unreadable)"]
                print(f"  {p.relative_to(path.parent)}  columns: {h}")
        return 0
    except DatasetError as exc:
        print(f"DATASET PROBLEM: {exc}")
        return 1
    except FileNotFoundError:
        print("File not found. Put your CSV in data/raw/ or pass the path as an argument.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
