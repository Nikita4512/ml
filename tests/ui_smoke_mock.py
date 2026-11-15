"""UI integration smoke test WITHOUT a browser.   Run:  python tests/ui_smoke_mock.py

Replaces `streamlit` and `plotly` with small checking fakes, then executes every page (and app.py)
against real pipeline output. It catches wrong imports, wrong column names passed to charts,
duplicate widget keys, backend/frontend mismatches and crashes in button handlers.
It can NOT judge how the pages look: that needs a real `streamlit run app.py`.
"""
import runpy
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------- fake plotly
class FakeFig:
    def __getattr__(self, name):
        return lambda *a, **k: self
    def to_html(self, *a, **k):
        return "<div></div>"


def _px(name):
    def f(df=None, *a, **kw):
        if isinstance(df, pd.DataFrame) and name != "imshow":
            for k in ("x", "y", "color", "size", "text", "hover_name", "hover_data", "error_y", "names", "values", "path"):
                v = kw.get(k)
                for col in (v if isinstance(v, (list, tuple)) else [v]):
                    if isinstance(col, str) and col not in df.columns:
                        raise ValueError(f"px.{name}: column '{col}' not in {list(df.columns)}")
        return FakeFig()
    return f


def install_fake_plotly():
    px, go, root = types.ModuleType("plotly.express"), types.ModuleType("plotly.graph_objects"), types.ModuleType("plotly")
    for n in ("bar", "scatter", "line", "histogram", "treemap", "imshow", "pie"):
        setattr(px, n, _px(n))
    go.Figure = lambda *a, **k: FakeFig()
    for n in ("Scatter", "Bar", "Heatmap", "Indicator", "Scatterpolar"):
        setattr(go, n, lambda *a, **k: dict(k))
    root.express, root.graph_objects = px, go
    sys.modules.update({"plotly": root, "plotly.express": px, "plotly.graph_objects": go})


# ----------------------------------------------------------------------------- fake streamlit
class Rerun(Exception):
    pass


class Ctx:
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(self._st, name)


class FakeSt:
    def __init__(self):
        self.session_state, self.clicks, self.keys, self.charts, self.errors = {}, set(), set(), 0, []
        self.sidebar = Ctx(self)

    def begin(self, clicks=()):
        self.clicks, self.keys, self.charts, self.errors = set(clicks), set(), 0, []

    def _key(self, key):
        if key is not None:
            assert key not in self.keys, f"DUPLICATE widget/chart key: {key}"
            self.keys.add(key)

    def __getattr__(self, name):
        def noop(*a, **k):
            if name == "error":
                self.errors.append(str(a[0]) if a else "")
            return None
        return noop

    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels): return [Ctx(self) for _ in labels]
    def expander(self, *a, **k): return Ctx(self)
    def spinner(self, *a, **k): return Ctx(self)
    def progress(self, *a, **k): return Ctx(self)
    def rerun(self): raise Rerun()

    def _val(self, key, default):
        self._key(key)
        return self.session_state.get(key, default) if key else default

    def selectbox(self, label, options, index=0, key=None, **k):
        options = list(options)
        return self._val(key, options[index] if options else None)
    def multiselect(self, label, options, default=None, key=None, **k): return self._val(key, list(default or []))
    def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None, **k):
        return self._val(key, value if value is not None else min_value)
    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, **k): return self._val(key, value)
    def checkbox(self, label, value=False, key=None, **k): return self._val(key, value)
    def radio(self, label, options, index=0, key=None, **k):
        options = list(options)
        return self._val(key, options[index] if options else None)
    def text_input(self, label, value="", key=None, **k): return self._val(key, value)
    def text_area(self, label, value="", key=None, **k): return self._val(key, value)
    def file_uploader(self, *a, key=None, **k): self._key(key); return None

    def button(self, label, key=None, on_click=None, args=(), **k):
        self._key(key)
        hit = (key in self.clicks) or (label in self.clicks)
        if hit and on_click:
            on_click(*args)
        return hit

    def download_button(self, label, data, file_name=None, mime=None, key=None, **k):
        self._key(key)
        assert isinstance(data, (bytes, str)) and len(data) > 0, f"empty download data for {label}"
        return False

    def plotly_chart(self, fig, key=None, **k):
        assert key is not None, "plotly_chart without key"
        self._key(key)
        self.charts += 1

    def dataframe(self, df, key=None, **k):
        assert isinstance(df, pd.DataFrame), "st.dataframe expects a DataFrame"
        if key is not None:
            self._key(key)


def main():
    install_fake_plotly()
    st = FakeSt()
    sys.modules["streamlit"] = st
    tmp = Path(tempfile.mkdtemp())

    from src import app_state as S
    from src.orchestrator import load_artifacts, run_pipeline
    from src.demo_data import ensure_demo_csv
    import views.upload as up_view
    # never write test output into the user's real folders
    S.DEMO_PROCESSED_DIR, S.DEMO_MODELS_DIR = tmp / "demo_out", tmp / "demo_models"
    up_view.PROCESSED_DIR, up_view.MODELS_DIR = tmp / "real_out", tmp / "real_models"

    csv = ensure_demo_csv()
    print("building demo artefacts (Mode A) ...")
    art_a = S.build_demo()
    narrow_csv = tmp / "narrow.csv"
    d = pd.read_csv(csv)
    ts = pd.to_datetime(d["listed_time"], unit="ms")
    d[ts < pd.Timestamp("2024-02-01")].to_csv(narrow_csv, index=False)
    art_b = run_pipeline(narrow_csv, tmp / "narrow_out", tmp / "narrow_models", source_kind="demo")
    print(f"Mode A art: mode={art_a['meta']['analysis_mode']}  Mode B art: mode={art_b['meta']['analysis_mode']}")
    assert art_a["meta"]["analysis_mode"] in ("A", "A_short") and art_b["meta"]["analysis_mode"] == "B"

    import importlib
    names = ["home", "upload", "demand", "trends", "models", "gap", "roadmap", "reports", "about"]
    mods = {n: importlib.import_module(f"views.{n}") for n in names}
    kind = S.DEMO
    failures = []

    def run(page, art, clicks=(), state=None, label="", expect_error=False):
        st.begin(clicks)
        st.session_state.update(state or {})
        try:
            mods[page].render(art, kind)
        except Rerun:
            pass
        except Exception as exc:
            import traceback
            failures.append((page, label, traceback.format_exc()))
            print(f"  FAIL {page} {label}: {type(exc).__name__}: {exc}")
            return
        if expect_error:
            if not st.errors:
                failures.append((page, label, "expected a friendly st.error but none was shown"))
                print(f"  FAIL {page} {label}: expected an error message")
            else:
                print(f"  ok   {page:8} {label:28} friendly error shown: {st.errors[0][:60]}")
            return
        if st.errors:
            failures.append((page, label, "st.error shown: " + st.errors[0]))
            print(f"  FAIL {page} {label}: st.error -> {st.errors[0][:150]}")
            return
        print(f"  ok   {page:8} {label:28} charts={st.charts}")

    for label, art in (("mode A", art_a), ("mode B", art_b)):
        print(f"--- pages with {label} artefacts")
        st.session_state.clear()
        for p in ("home", "demand", "trends", "models", "about"):
            run(p, art, label=label)
        run("demand", art, state={f"dm_roles_{kind}": ["Data Scientist"], f"dm_levels_{kind}": ["junior"],
                                  f"dm_cats_{kind}": ["Data & Analytics"]}, label=label + " + filters")
        run("models", art, clicks={f"ml_pred_{kind}"}, state={f"ml_text_{kind}": "We need python, pandas and machine learning"},
            label=label + " + predict")
        run("gap", art, label=label + " (before analyse)")
        run("gap", art, clicks={f"gap_go_{kind}"}, state={f"gap_ms_{kind}": ["Python", "SQL"], f"gap_txt_{kind}": "excel, tableau, juggling"},
            label=label + " + analyse")
        run("gap", art, label=label + " (result shown)")
        run("roadmap", art, label=label)
        run("reports", art, clicks={"rep_pdf_btn", "rep_xlsx_btn", "rep_html_btn"}, label=label + " + generate")
        run("reports", art, label=label + " (downloads)")
        for k, sig in (("report_pdf", b"%PDF"), ("report_excel", b"PK"), ("report_html", b"<html")):
            blob = st.session_state.get(k)
            assert blob and blob[:len(sig)] == sig, f"{k} not generated correctly"

    print("--- empty-state pages (no data)")
    st.session_state.clear()
    for p in ("home", "demand", "trends", "models", "gap", "roadmap", "reports"):
        run(p, None, label="no data")

    print("--- upload page: path -> preview -> run")
    st.session_state.clear()
    run("upload", art_a, state={"up_path": str(csv)}, label="path set")
    run("upload", art_a, clicks={"up_preview"}, state={"up_path": str(csv)}, label="preview")
    run("upload", art_a, clicks={"up_run"}, state={"up_path": str(csv), "up_maxrows": 3000, "up_train": False},
        label="run (3000 rows, no models)")
    assert (tmp / "real_out" / "meta.json").exists(), "upload run did not write results"
    run("upload", art_a, state={"up_path": str(tmp / "missing.csv")}, label="missing file", expect_error=True)
    bad = tmp / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    run("upload", art_a, clicks={"up_run"}, state={"up_path": str(bad)}, label="CSV without text columns", expect_error=True)

    print("--- whole app.py (navigation through every page)")
    S.available = lambda k: k == S.DEMO or (tmp / "real_out" / "meta.json").exists()
    S.dirs_for = lambda k: (tmp / "demo_out", tmp / "demo_models") if k == S.DEMO else (tmp / "real_out", tmp / "real_models")
    import src.orchestrator as orch
    S.artifacts_exist = orch.artifacts_exist
    st.session_state.clear()
    for page in ["Home", "Dataset & Upload", "Skill Demand", "Emerging & Declining", "ML Models", "Skill Gap Analyzer",
                 "Recommendations & Roadmap", "Reports & Downloads", "About & Viva Guide"]:
        st.begin()
        st.session_state["page"] = page
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            ok = not st.errors
        except Rerun:
            ok = True
        except Exception as exc:
            ok = False
            failures.append(("app.py", page, repr(exc)))
        print(f"  {'ok  ' if ok else 'FAIL'} app.py -> {page}")
        if not ok and st.errors:
            failures.append(("app.py", page, st.errors[0]))

    print()
    if failures:
        for p, l, tb in failures:
            print(f"FAILED: {p} [{l}]\n{tb}\n")
        sys.exit(1)
    print("ALL UI SMOKE CHECKS PASSED (fake Streamlit/Plotly; see README for what still needs a real browser)")


if __name__ == "__main__":
    main()
