import sys
from pathlib import Path

import streamlit as st

# let the app find src/selector.py
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from selector import load_all, filter_materials, recommend

st.set_page_config(page_title="Material selector", layout="wide")


@st.cache_data
def get_all():
    return load_all()


all_df = get_all()
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
            st.dataframe(table, hide_index=True, use_container_width=True)

with tab2:
    st.info("Cluster map: added in the next step.")

with tab3:
    st.info("Find substitutes: added in a later step.")