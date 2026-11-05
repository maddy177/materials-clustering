"""Computes the 2D PCA position of each clustered material and saves it for the app."""
import pandas as pd
from sklearn.decomposition import PCA
from selector import load_scaled, DATA_DIR

X, meta = load_scaled()
pca = PCA(n_components=2, random_state=42).fit(X)
P = pca.transform(X)

out = pd.DataFrame({"label": X.index, "PC1": P[:, 0], "PC2": P[:, 1]})
out.to_csv(DATA_DIR / "pca_coords.csv", index=False)

print("Saved", len(out), "points")
print("Variance explained:", (pca.explained_variance_ratio_ * 100).round(1), "%")