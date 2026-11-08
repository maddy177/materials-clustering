import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# let the app find src/selector.py
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from selector import (load_all, load_scaled, filter_materials, recommend,
                      similar_to, CORE, DATA_DIR)

st.set_page_config(page_title="Material selector", layout="wide")


@st.cache_data
def get_all():
    return load_all()


@st.cache_data
def get_scaled():
    return load_scaled()


@st.cache_data
def get_pca():
    return pd.read_csv(DATA_DIR / "pca_coords.csv")


all_df = get_all()
X, meta = get_scaled()
all_families = sorted(all_df["family"].dropna().unique())

st.title("Material selector")
st.write("A first-pass screening tool for 1,256 structural metals. "
         "Set your requirements on the left and the matching materials appear below.")

st.sidebar.header("Requirements")
min_tensile = st.sidebar.slider("Minimum tensile strength (MPa)", 0, 2200, 0, step=10)
min_elong = st.sidebar.slider("Minimum elongation (%)", 0, 70, 0)
max_density = st.sidebar.slider("Maximum density (g/cm3)", 1.5, 9.0, 9.0, step=0.1)
families = st.sidebar.multiselect("Families", all_families, default=all_families)
rank_by = st.sidebar.selectbox(
    "Rank by",
    ["specific_strength", "yield_strength_MPa", "tensile_strength_MPa",
     "elongation_pct", "youngs_modulus_GPa"])
top = st.sidebar.slider("Rows to show", 5, 50, 10)

# a limit left at its starting value means "no limit"
constraints = {}
if min_tensile > 0:
    constraints["tensile_strength_MPa"] = (min_tensile, None)
if min_elong > 0:
    constraints["elongation_pct"] = (min_elong, None)
if max_density < 9.0:
    constraints["density_g_cm3"] = (None, max_density)

tab1, tab2, tab3 = st.tabs(["Find materials", "Cluster map", "Find substitutes"])

with tab1:
    if not families:
        st.warning("Select at least one family.")
    else:
        matches = filter_materials(all_df, constraints, families)
        st.metric("Matching materials", f"{len(matches)} of {len(all_df)}")
        if matches.empty:
            st.warning("No material satisfies all requirements. Relax one.")
        else:
            st.write("Matches by cluster")
            st.bar_chart(matches["cluster_name"].value_counts())
            st.write(f"Top {min(top, len(matches))} matches, ranked by {rank_by}")
            table = recommend(all_df, constraints, rank_by=rank_by, top=top, families=families)
            st.dataframe(table, hide_index=True)

with tab2:
    if not families:
        st.warning("Select at least one family.")
    else:
        pca = get_pca()
        matches = filter_materials(all_df, constraints, families)
        # grey-iron labels repeat, so keep one row per label before looking up clusters
        cluster_of = all_df.drop_duplicates("label").set_index("label")["cluster_name"]
        hit = pca["label"].isin(matches["label"])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(pca["PC1"], pca["PC2"], s=10, color="lightgrey", label="all materials")
        shown = pca[hit].assign(cluster=pca.loc[hit, "label"].map(cluster_of))
        for name, grp in shown.groupby("cluster"):
            ax.scatter(grp["PC1"], grp["PC2"], s=22, label=f"{name} ({len(grp)})")
        ax.set_xlabel("PC1 (61% of variance): overall heft, strength + stiffness + density")
        ax.set_ylabel("PC2 (28% of variance): ductility versus strength")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        grey_matches = int(matches["cluster_name"].str.startswith("Grey", na=False).sum())
        st.write(f"{int(hit.sum())} matching materials are drawn. "
                 f"{grey_matches} matching grey-iron rows are not on the map because they were not clustered.")

with tab3:
    st.write("Pick a material to see which others have the closest overall properties. "
             "Distance is measured in the scaled property space used for clustering, "
             "so a smaller distance means more similar. It has no units.")

    labels = list(X.index)
    default = labels.index("Steel SAE 4140 | normalized") if "Steel SAE 4140 | normalized" in labels else 0
    chosen = st.selectbox("Material (click and type to search)", labels, index=default)
    n_sub = st.slider("Number of substitutes", 3, 10, 5)

    st.write("Selected material")
    st.dataframe(meta.loc[[chosen], ["family", "cluster_name"] + CORE])

    subs = similar_to(X, meta, chosen, n=n_sub)
    subs["different family"] = subs["family"] != meta.loc[chosen, "family"]
    subs["elongation imputed"] = meta.loc[subs.index, "elongation_imputed"].astype(bool)
    st.write(f"Closest {n_sub} materials")
    st.dataframe(subs)

    n_cross = int(subs["different family"].sum())
    if n_cross:
        st.write(f"{n_cross} of these belong to a different family. They are alternatives "
                 "that behave similarly on the five mechanical properties, "
                 "but check chemistry, corrosion and temperature behaviour before using them.")