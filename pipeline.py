"""Command-line entry point: process the big CSV ONCE, then use the dashboard.

Examples
    python pipeline.py                                  # auto-finds the CSV in data/raw/
    python pipeline.py --postings data/raw/postings.csv
    python pipeline.py --max-rows 20000                 # quick test run
    python pipeline.py --no-models                      # skip ML training (faster)
    python pipeline.py --demo data/sample/demo_postings.csv   # label results as DEMO data
"""
import argparse
import sys
import time

from src.config import CHUNK_SIZE, PROCESSED_DIR, RAW_DIR
from src.data_loader import DatasetError, find_postings_file
from src.orchestrator import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Skill obsolescence pipeline")
    ap.add_argument("--postings", help="Path to the postings CSV (default: auto-detect in data/raw)")
    ap.add_argument("--demo", help="Path to a DEMO/synthetic CSV (results are labelled DEMO)")
    ap.add_argument("--max-rows", type=int, default=None, help="Only read the first N rows")
    ap.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    ap.add_argument("--no-models", action="store_true", help="Skip ML training")
    ap.add_argument("--out", default=str(PROCESSED_DIR), help="Output folder for processed artefacts")
    args = ap.parse_args()

    try:
        if args.demo:
            path, kind = args.demo, "demo"
        else:
            path, kind = (args.postings or find_postings_file(RAW_DIR)), "real"
        print(f"Source file : {path}\nSource type : {kind.upper()}")
        t0 = time.time()
        last = {"msg": ""}

        def progress(msg, frac):
            if msg != last["msg"]:
                print(f"[{frac * 100:5.1f}%] {msg}")
                last["msg"] = msg

        art = run_pipeline(path, out_dir=args.out, source_kind=kind, max_rows=args.max_rows,
                           chunksize=args.chunksize, train_models=not args.no_models, progress=progress)
    except DatasetError as exc:
        print(f"\nDATASET PROBLEM: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"\nFILE NOT FOUND: {exc}")
        return 1
    except Exception as exc:                                   # unexpected: show a readable message
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        raise

    m = art["meta"]
    print("\n==== SUMMARY (all values computed from your data) ====")
    print(f"Postings analysed     : {m['rows']:,}")
    print(f"Skills detected       : {m['unique_skills_found']}")
    print(f"Date range            : {m['date_min']} -> {m['date_max']}")
    print(f"Temporal analysis mode: {m['analysis_mode']}  ({'; '.join(m['temporal']['reasons'])})")
    for name, res in art["metrics"].items():
        if res.get("status") == "ok":
            if name == "role_classifier":
                print(f"Role classifier       : {res['best_model']}  test macro-F1={res['test_macro_f1']:.3f} "
                      f"(baseline {res['baseline_macro_f1']:.3f}), accuracy={res['test_accuracy']:.3f}")
            else:
                tm, bm = res["test_metrics"], res["baseline_metrics"]
                print(f"Salary model          : {res['best_model']}  MAE={tm['mae_usd']:,.0f} (baseline {bm['mae_usd']:,.0f}), R2(log)={tm['r2_log']:.3f}")
        else:
            print(f"{name:22}: {res.get('status')} - {res.get('reason') or res.get('error')}")
    print(f"\nFinished in {time.time() - t0:.0f}s. Next: streamlit run app.py  (app comes in the next message)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
