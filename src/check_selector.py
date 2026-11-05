"""Checks that the selector gives the same answers as notebook 08."""
from selector import load_all, load_scaled, filter_materials, recommend, similar_to

all_df = load_all()
X, meta = load_scaled()
print("Loaded:", len(all_df), "materials,", len(X), "clustered")

scenarios = {
    "A lightweight": ({"density_g_cm3": (None, 3.0), "tensile_strength_MPa": (300, None)}, 65),
    "B shaft":       ({"tensile_strength_MPa": (900, None), "elongation_pct": (10, None)}, 99),
    "C formed part": ({"elongation_pct": (30, None), "tensile_strength_MPa": (300, None)}, 157),
}
ok = True
for name, (constraints, expected) in scenarios.items():
    n = len(filter_materials(all_df, constraints))
    status = "OK" if n == expected else "MISMATCH"
    ok = ok and n == expected
    print(f"{name}: {n} matches (expected {expected}) -> {status}")

top = recommend(all_df, scenarios["B shaft"][0], rank_by="yield_strength_MPa", top=3)
print("\nTop 3 for the shaft scenario:")
print(top[["label", "tensile_strength_MPa", "elongation_pct"]].to_string(index=False))

sub = similar_to(X, meta, "Steel SAE 4140 | normalized", n=5)
print("\nFive materials closest to Steel SAE 4140 | normalized:")
print(sub[["family", "distance"]].to_string())

print("\nALL CHECKS PASSED" if ok else "\nSOMETHING DOES NOT MATCH")