"""Emerging / declining skills, forecasts and Skill Market Risk (honest about data limits)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui_components as ui
from src.config import OBSOLESCENCE_WEIGHTS
from src.frontend_logic import MODE_LABEL, STATUS_COLOR, history_forecast
from src.reports import df_to_csv_bytes
from src.trends import RISK_SCORE_METHODOLOGY


def _mode_banner(meta):
    mode, t = meta["analysis_mode"], meta.get("temporal", {})
    text = f"**{MODE_LABEL.get(mode, mode)}**  \n" + "  \n".join(t.get("reasons", []))
    (st.success if mode == "A" else st.warning)(text)


def render(art, kind):
    ui.page_header("Emerging & Declining Skills", "Time trends, forecasts and Skill Market Risk - only where the data supports it")
    if art is None:
        ui.no_data(kind)
        return
    meta = art["meta"]
    ui.source_banner(meta)
    _mode_banner(meta)
    mode = meta["analysis_mode"]
    tt, series = art.get("trend_table"), art.get("trend_series")
    freq = meta.get("temporal", {}).get("freq")

    tabs = st.tabs(["Trends", "Forecast", "Skill Market Risk", "Junior vs senior signal", "Legacy watchlist"])

    # ------------------------------------------------------------------ Trends
    with tabs[0]:
        tc = art.get("temporal_counts")
        if tc is not None and len(tc):
            ui.show_fig(ui.style_fig(px.bar(tc, x="period", y="postings"), 260, "Postings per month (coverage)"),
                        key=f"tr_cov_{kind}")
        if mode == "B" or tt is None or tt.empty or series is None:
            st.warning("Time trends are **disabled** because the dataset does not have enough dated postings "
                       "(see the reasoning above). No trend is shown rather than inventing one. "
                       "Use the 'Junior vs senior signal' tab for a weak cross-sectional proxy.")
        else:
            if mode == "A_short":
                st.warning("Weekly trends over a short window: evidence is capped at WEAK. They may reflect how the data "
                           "was collected, not real market change.")
            c1, c2 = st.columns(2)
            status_sel = c1.multiselect("Status", ["Emerging", "Declining", "Stable", "Insufficient volume"],
                                        default=["Emerging", "Declining"], key=f"tr_status_{kind}")
            ev_sel = c2.multiselect("Evidence", ["Strong", "Moderate", "Weak"], default=["Strong", "Moderate", "Weak"],
                                    key=f"tr_ev_{kind}")
            view = tt[tt["status"].isin(status_sel) & tt["evidence"].isin(ev_sel)].copy()
            view["change_pct_per_period"] = view["rel_change_per_period"] * 100
            if view.empty:
                st.info("No skills match these filters.")
            else:
                top = pd.concat([view.head(10), view.tail(10)]).drop_duplicates("skill").iloc[::-1]
                fig = px.bar(top, x="change_pct_per_period", y="skill", color="status", orientation="h",
                             color_discrete_map=STATUS_COLOR, labels={"change_pct_per_period": "Relative change per period (%)", "skill": ""})
                ui.show_fig(ui.style_fig(fig, max(380, 24 * len(top) + 120), "Fastest rising and falling skills"),
                            key=f"tr_bar_{kind}")
                ui.show_df(view[["skill", "status", "evidence", "avg_share_pct", "change_pct_per_period", "r2", "q_value", "n_periods"]],
                           key=f"tr_tbl_{kind}", height=300)
                st.download_button("Download trend table (CSV)", df_to_csv_bytes(view), "skill_trends.csv", "text/csv",
                                   key=f"tr_dl_{kind}")
            st.markdown("**Compare skill share over time**")
            default = [s for s in (view["skill"].head(2).tolist() + view["skill"].tail(2).tolist())] if len(view) else []
            picks = st.multiselect("Skills", list(series.columns), default=list(dict.fromkeys(default)), key=f"tr_pick_{kind}")
            if picks:
                fig = go.Figure()
                for s in picks:
                    fig.add_trace(go.Scatter(x=series.index, y=series[s] * 100, mode="lines+markers", name=s))
                fig.update_yaxes(title="% of postings")
                ui.show_fig(ui.style_fig(fig, 400, "Skill share per period"), key=f"tr_line_{kind}")

    # ---------------------------------------------------------------- Forecast
    with tabs[1]:
        if mode != "A" or series is None:
            st.warning("Forecasting is **disabled**: it needs at least 6 months with enough postings. "
                       "This protects you from presenting a fabricated forecast.")
        else:
            bt = meta.get("backtest", {})
            if bt.get("status") == "ok":
                m = bt["mae_pct_points"]
                ui.kpi_row([("Back-test: linear trend", f"{m['linear_trend']:.3f} pp", "mean abs. error"),
                            ("Back-test: naive last value", f"{m['naive_last_value']:.3f} pp", "baseline"),
                            ("Back-test: mean of last 3", f"{m['mean_last_3']:.3f} pp", "baseline"),
                            ("Linear beats naive?", "Yes" if bt["linear_beats_naive"] else "No",
                             f"hold-out {bt['holdout_periods']} periods, {bt['n_skills']} skills")])
                if not bt["linear_beats_naive"]:
                    st.warning("On this data the linear forecast is NOT better than simply repeating the last value. "
                               "Treat the forecast below with caution.")
            else:
                st.info(f"Back-test skipped: {bt.get('reason', 'not enough periods')}")
            c1, c2 = st.columns([3, 1])
            skill = c1.selectbox("Skill to forecast", list(series.columns), key=f"fc_skill_{kind}")
            horizon = c2.slider("Periods ahead", 1, 6, 3, key=f"fc_h_{kind}")
            hist, fc = history_forecast(art, skill, horizon)
            if hist is None or fc.empty:
                st.info("Not enough history to forecast this skill.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="History"))
                fig.add_trace(go.Scatter(x=fc["period"], y=fc["upper_pct"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=fc["period"], y=fc["lower_pct"], mode="lines", line=dict(width=0), fill="tonexty",
                                         name="90% interval", fillcolor="rgba(167,139,250,0.25)"))
                fig.add_trace(go.Scatter(x=fc["period"], y=fc["forecast_pct"], mode="lines+markers", name="Forecast",
                                         line=dict(dash="dash")))
                fig.update_yaxes(title="% of postings")
                ui.show_fig(ui.style_fig(fig, 420, f"{skill}: history and linear-trend forecast"), key=f"fc_chart_{kind}")
                ui.show_df(fc, key=f"fc_tbl_{kind}")
                st.caption("Method: ordinary least-squares line through the observed monthly shares, with a prediction interval "
                           "from the residuals. It is only a trend extrapolation, not a market prediction.")

    # ---------------------------------------------------------------- Obsolescence
    with tabs[2]:
        ob = art.get("obsolescence")
        if ob is None or ob.empty:
            st.info("Not enough skills (>= 30 postings) to score obsolescence risk.")
        else:
            st.markdown("**Risk score (0-100)** = weighted mix of: " + ", ".join(f"{k} {v:.0%}" for k, v in OBSOLESCENCE_WEIGHTS.items())
                        + ". Components that are unavailable for this dataset are dropped and the rest re-weighted "
                        "(see `components_used`). The legacy flag is shown but is not part of the score.")
            with st.expander("Methodology and limitations (click to expand)"):
                st.text(RISK_SCORE_METHODOLOGY)
                st.caption("WARNING: This is a HEURISTIC INDICATOR, not a probability that a skill will become obsolete.")
            c1, c2 = st.columns(2)
            lv = c1.multiselect("Risk level", ["High", "Moderate", "Low"], default=["High", "Moderate"], key=f"ob_lv_{kind}")
            cats = c2.multiselect("Category", sorted(ob["category"].unique()), key=f"ob_cat_{kind}")
            v = ob[ob["risk_level"].astype(str).isin(lv)]
            if cats:
                v = v[v["category"].isin(cats)]
            if v.empty:
                st.info("No skills match these filters.")
            else:
                fig = px.scatter(v, x="share", y="risk_score", size="postings", color="risk_level", hover_name="skill",
                                 color_discrete_map={"High": "#F87171", "Moderate": "#FBBF24", "Low": "#34D399"},
                                 labels={"share": "Share of postings", "risk_score": "Skill Market Risk Score"})
                ui.show_fig(ui.style_fig(fig, 440, "Demand vs Skill Market Risk Score"), key=f"ob_sc_{kind}")
                display_cols = ["skill", "category", "postings", "risk_score", "risk_level", "confidence_level", "evidence", "components_used", "legacy_watchlist_flag"]
                show_cols = [c for c in display_cols if c in v.columns]
                ui.show_df(v[show_cols], key=f"ob_tbl_{kind}", height=320)
                st.download_button("Download risk table (CSV)", df_to_csv_bytes(v), "obsolescence_risk.csv", "text/csv",
                                   key=f"ob_dl_{kind}")

    # ---------------------------------------------------------------- Cross-sectional
    with tabs[3]:
        cs = art.get("cross_sectional")
        st.info("This is NOT a time trend. It compares how often a skill appears in junior vs senior postings. "
                "Junior-leaning skills are a weak hint of newer demand; senior-leaning ones a weak hint of established "
                "or legacy skills. Evidence is always WEAK.")
        if cs is None or cs.empty:
            st.warning(f"Not available: {meta.get('seniority_note', 'not enough labelled junior/senior postings')}.")
        else:
            top = pd.concat([cs.head(10), cs.tail(10)]).drop_duplicates("skill").iloc[::-1]
            fig = px.bar(top, x="log_ratio", y="skill", color="signal", orientation="h",
                         labels={"log_ratio": "log(junior share / senior share)", "skill": ""})
            ui.show_fig(ui.style_fig(fig, max(380, 24 * len(top) + 120), "Junior-leaning vs senior-leaning skills"),
                        key=f"cs_bar_{kind}")
            ui.show_df(cs, key=f"cs_tbl_{kind}", height=300)

    # ---------------------------------------------------------------- Legacy
    with tabs[4]:
        lg = art.get("legacy")
        st.warning("KNOWLEDGE-BASED list: these technologies are widely regarded as legacy. The reason text is NOT derived "
                   "from your dataset; only the observed counts are.")
        if lg is not None and len(lg):
            ui.show_df(lg, key=f"lg_tbl_{kind}", height=420)
