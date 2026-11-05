"""Material selection helpers, shared by the notebook and the Streamlit page."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# data/processed, found relative to this file, so it works from any folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CORE = ["tensile_strength_MPa", "yield_strength_MPa", "elongation_pct",
        "youngs_modulus_GPa", "density_g_cm3"]

SHOW = ["label", "family", "cluster_name", "tensile_strength_MPa", "yield_strength_MPa",
        "elongation_pct", "youngs_modulus_GPa", "density_g_cm3", "specific_strength"]


def load_all():
    """All 1,256 materials (1,235 clustered + 21 grey-iron rows kept aside)."""
    all_df = pd.read_csv(DATA_DIR / "materials_clustered_all.csv")
    all_df["elongation_assumed"] = all_df["elongation_assumed"].fillna(False).astype(bool)
    # grey iron: the 0.5% was an assumption, so show it as "not reported"
    all_df.loc[all_df["elongation_assumed"], "elongation_pct"] = np.nan
    all_df["specific_strength"] = all_df["tensile_strength_MPa"] / all_df["density_g_cm3"]
    all_df["specific_stiffness"] = all_df["youngs_modulus_GPa"] / all_df["density_g_cm3"]
    return all_df


def load_scaled():
    """Scaled properties X and matching metadata for the 1,235 clustered materials."""
    X = pd.read_csv(DATA_DIR / "materials_scaled.csv", index_col=0)
    meta = pd.read_csv(DATA_DIR / "materials_clustered.csv", index_col=0).loc[X.index]
    return X, meta


def filter_materials(all_df, constraints, families=None):
    """Rows that satisfy every constraint. constraints = {column: (min, max)}, None = no limit."""
    mask = pd.Series(True, index=all_df.index)
    for col, (lo, hi) in constraints.items():
        if lo is not None:
            mask &= all_df[col] >= lo
        if hi is not None:
            mask &= all_df[col] <= hi
    if families:
        mask &= all_df["family"].isin(families)
    return all_df[mask]


def recommend(all_df, constraints, rank_by="specific_strength", ascending=False,
              top=8, families=None):
    """The best `top` matches, ranked by one property."""
    matches = filter_materials(all_df, constraints, families)
    return matches.sort_values(rank_by, ascending=ascending).head(top)[SHOW].round(1).reset_index(drop=True)


def similar_to(X, meta, label, n=5):
    """The n materials closest to `label` in scaled property space."""
    nn = NearestNeighbors(n_neighbors=n + 1).fit(X)
    i = X.index.get_loc(label)
    dist, idx = nn.kneighbors(X.iloc[[i]], n_neighbors=n + 1)
    out = meta.loc[X.index[idx[0][1:]], ["family", "cluster_name"] + CORE].copy()
    out["distance"] = dist[0][1:].round(2)
    return out