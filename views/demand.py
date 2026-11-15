"""Skill extraction and demand analysis page (filters recompute results live from the skill matrix)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analysis
from src import ui_components as ui
from src.frontend_logic import filter_mask, jaccard_from_counts
from src.reports import df_to_csv_bytes
from src.skills_taxonomy import SKILL_CATEGORY


def render(art, kind):
    ui.page_header("Skill Extraction & Demand Analysis", "How often each technical skill appears in job postings")
    if art is None:
        ui.no_data(kind)
        return
    ui.source_banner(art["meta"])
    clean, X, skills = art["clean"], art["matrix"], art["skills"]

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    roles_all = sorted(r for r in clean["role"].unique() if r != "Other")
    roles = f1.multiselect("Role family", roles_all, key=f"dm_roles_{kind}")
    levels = f2.multiselect("Experience group", sorted(clean["level_group"].unique()), key=f"dm_levels_{kind}")
    cat_sel = f3.multiselect("Skill category", sorted(set(SKILL_CATEGORY.values())), key=f"dm_cats_{kind}")
    top_n = f4.slider("Top N", 5, 50, 20, key=f"dm_topn_{kind}")

    mask = filter_mask(clean, roles, levels)
    n = int(mask.sum())
    if n == 0:
        st.warning("No postings match these filters. Remove a filter to continue.")
        return
    Xs = X[mask]
    demand = analysis.skill_demand(Xs, skills, SKILL_CATEGORY, min_postings=1)
    demand = demand[demand["postings"] > 0].copy()
    demand["share_pct"] = demand["share"] * 100
    view = demand[demand["category"].isin(cat_sel)] if cat_sel else demand
    per_post = np.asarray(Xs.sum(axis=1)).ravel()

    ui.kpi_row([("Postings in filter", f"{n:,}", f"of {len(clean):,} total"),
                ("Distinct skills", f"{len(demand):,}", "found in this selection"),
                ("Avg skills / posting", f"{per_post.mean():.2f}", "mean count"),
                ("Postings with a skill", f"{(per_post > 0).mean() * 100:.1f}%", "at least one match")])
    st.write("")

    tabs = st.tabs(["Top skills", "Categories", "Co-occurrence", "By role", "Salary link", "Skill discovery", "Table & export"])

    with tabs[0]:
        top = view.head(top_n).iloc[::-1]
        if top.empty:
            st.info("No skills in this selection.")
        else:
            fig = px.bar(top, x="share_pct", y="skill", color="category", orientation="h", text="postings",
                         labels={"share_pct": "% of postings", "skill": ""})
            ui.show_fig(ui.style_fig(fig, max(380, 22 * len(top) + 120), f"Top {len(top)} skills"), key=f"dm_top_{kind}")
        c1, c2 = st.columns(2)
        hist = px.histogram(pd.DataFrame({"skills_per_posting": per_post}), x="skills_per_posting", nbins=20)
        c1.markdown("**How many skills does a typical posting list?**")
        with c1:
            ui.show_fig(ui.style_fig(hist, 300), key=f"dm_hist_{kind}")
        c2.markdown("**Reading the chart**")
        c2.write("Bar length = share of the selected postings that mention the skill. Colours are skill categories "
                 "from the curated taxonomy. Counts come straight from the extracted skill matrix.")

    with tabs[1]:
        cat = analysis.category_demand(Xs, skills, SKILL_CATEGORY).assign(share_pct=lambda d: d["share"] * 100)
        c1, c2 = st.columns(2)
        with c1:
            ui.show_fig(ui.style_fig(px.bar(cat.iloc[::-1], x="share_pct", y="category", orientation="h",
                                            labels={"share_pct": "% of postings with >= 1 skill in category", "category": ""}),
                                     420, "Postings per skill category"), key=f"dm_cat_{kind}")
        tree = view.head(120)
        with c2:
            if len(tree):
                ui.show_fig(ui.style_fig(px.treemap(tree, path=["category", "skill"], values="postings"), 420,
                                         "Skill landscape (area = postings)"), key=f"dm_tree_{kind}")

    with tabs[2]:
        k = st.slider("Number of skills in the matrix", 10, 40, 25, key=f"dm_cooc_k_{kind}")
        cooc = analysis.cooccurrence(Xs, skills, top_n=k)
        if len(cooc) < 3:
            st.info("Not enough skills for a co-occurrence view.")
        else:
            jac = jaccard_from_counts(cooc)
            ui.show_fig(ui.style_fig(px.imshow(jac, color_continuous_scale=ui.PURPLE_SCALE, aspect="auto"), 560,
                                     "Skill co-occurrence (Jaccard similarity)"), key=f"dm_cooc_{kind}")
            arr, names, pairs = jac.to_numpy(), list(jac.index), []
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    pairs.append((names[a], names[b], float(arr[a, b]), int(cooc.iloc[a, b])))
            pdf = pd.DataFrame(pairs, columns=["skill_a", "skill_b", "jaccard", "postings_together"]) \
                .sort_values("jaccard", ascending=False).head(15)
            st.markdown("**Strongest skill pairs** (skills employers ask for together)")
            ui.show_df(pdf, key=f"dm_pairs_{kind}")

    with tabs[3]:
        rp = art.get("role_profile")
        if rp is None or rp.empty:
            st.info("No role has enough postings for a profile (title rules matched too few tech roles).")
        else:
            m = st.slider("Skills shown in heatmap", 10, 40, 20, key=f"dm_rp_k_{kind}")
            top_sk = rp.mean(axis=0).sort_values(ascending=False).head(m).index
            heat = (rp[top_sk] * 100).round(1)
            ui.show_fig(ui.style_fig(px.imshow(heat, color_continuous_scale=ui.PURPLE_SCALE, aspect="auto", text_auto=".0f"),
                                     max(360, 40 * len(heat) + 140), "% of each role's postings mentioning the skill"),
                        key=f"dm_rp_{kind}")
            role = st.selectbox("Inspect one role", list(rp.index), key=f"dm_rp_role_{kind}")
            one = (rp.loc[role].sort_values(ascending=False).head(15) * 100).rename("share_pct").rename_axis("skill").reset_index()
            ui.show_fig(ui.style_fig(px.bar(one.iloc[::-1], x="share_pct", y="skill", orientation="h"), 420,
                                     f"Most requested skills: {role}"), key=f"dm_rp_one_{kind}")

    with tabs[4]:
        sp = art.get("salary_premium")
        if sp is None or sp.empty:
            st.info("Not enough postings with salary information to compare skills. "
                    f"({art['meta'].get('salary_premium_note', '')})")
        else:
            st.caption("Median salary of postings WITH a skill vs WITHOUT it. This is an association, not proof that "
                       "the skill causes higher pay.")
            sp = sp.assign(significant=np.where(sp["p_value"] < 0.05, "p < 0.05", "not significant"))
            top = pd.concat([sp.head(12), sp.tail(5)]).drop_duplicates("skill").iloc[::-1]
            ui.show_fig(ui.style_fig(px.bar(top, x="premium_pct", y="skill", color="significant", orientation="h",
                                            labels={"premium_pct": "Median salary difference (%)", "skill": ""}),
                                     520, "Salary association by skill"), key=f"dm_sal_{kind}")
            ui.show_df(sp, key=f"dm_sal_tbl_{kind}", height=300)

    with tabs[5]:
        cand = art.get("candidate_skills")
        st.write("TF-IDF finds frequent, distinctive terms that are **not** in the skill dictionary. "
                 "They are suggestions: review them and add real skills to `src/skills_taxonomy.py`.")
        if cand is None or cand.empty:
            st.info("No candidate terms were found.")
        else:
            ui.show_df(cand, key=f"dm_cand_{kind}", height=420)

    with tabs[6]:
        out = view.drop(columns=["share"]).rename(columns={"share_pct": "share_of_postings_pct"})
        ui.show_df(out, key=f"dm_tbl_{kind}", height=420)
        st.download_button("Download this table (CSV)", df_to_csv_bytes(out), "skill_demand_filtered.csv", "text/csv",
                           key=f"dm_dl_{kind}")
