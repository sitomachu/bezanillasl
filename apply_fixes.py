#!/usr/bin/env python3
"""
Aplica los 2 fixes a 53_boost_sale_optuna.ipynb y 53_boost_rent.ipynb:
  Fix 1 — Eliminar ratio_dormitorios_superficie y ratio_banos_superficie de BASE_FEATURES
  Fix 2 — precio_m2_municipio_media calculada solo desde train (sale) /
           desde datos de VENTA por municipio (rent, sin leakage del target)
"""
import json, uuid, re
from pathlib import Path

ROOT = Path("/Users/sitomachucas/Documents/BezanillaSL")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_nb(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_nb(nb, path):
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

def src(cell):
    return "".join(cell["source"])

def set_src(cell, new_src):
    cell["source"] = new_src

# ═════════════════════════════════════════════════════════════════════════════
# 53_boost_sale_optuna.ipynb
# ═════════════════════════════════════════════════════════════════════════════
SALE_PATH = ROOT / "notebooks/05_ML/53_boost_sale_optuna.ipynb"
nb = load_nb(SALE_PATH)

# ── Fix 1 + Fix 2 en Cell 7 (BASE_FEATURES + build_X + llamada final) ────────
# Localizar cell 7
assert nb["cells"][7]["cell_type"] == "code"
old7 = src(nb["cells"][7])

# a) Quitar ratio features de BASE_FEATURES
new7 = old7.replace(
    '    "ratio_dormitorios_superficie",\n    "ratio_banos_superficie",\n',
    ""
)

# b) Añadir comentario en BASE_FEATURES para precio_m2_municipio_media
new7 = new7.replace(
    '    "precio_m2_municipio_media",',
    '    "precio_m2_municipio_media",  # se recalcula train-only justo abajo'
)

# c) Reemplazar el bloque final de cell 7 (X, feats = build_X / y = / print)
OLD_END_7 = (
    "X, feats = build_X(df)\n"
    "y = df[TARGET_COL].values\n"
    'print(f"Features ({len(feats)}):")\n'
    "print(feats)"
)
NEW_END_7 = """\
# ── Fix 2: precio_m2_municipio_media solo desde train (evita leakage) ─────────
_mun_oh = [c for c in df.columns if c.startswith("municipio_")]
df["_mun"] = df[_mun_oh].idxmax(axis=1).str.replace("municipio_", "")
_idx_tr, _idx_te = train_test_split(
    df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_mun_means = df.loc[_idx_tr].groupby("_mun")["precio_m2"].mean()
_global_mean = float(df.loc[_idx_tr, "precio_m2"].mean())
df["precio_m2_municipio_media"] = (
    df["_mun"].map(_mun_means).fillna(_global_mean)
)
df = df.drop(columns=["_mun"])

X, feats = build_X(df)
y = df[TARGET_COL].values
print(f"Features ({len(feats)}):")
print(feats)"""

new7 = new7.replace(OLD_END_7, NEW_END_7)
assert new7 != old7, "Cell 7 de sale no cambió — revisar marcadores"
set_src(nb["cells"][7], new7)

# ── Fix Cell 10 (split): usar _idx_tr / _idx_te precalculados ─────────────────
assert nb["cells"][10]["cell_type"] == "code"
old10 = src(nb["cells"][10])
NEW_10 = """\
# Split consistente con los índices usados para calcular precio_m2_municipio_media
X_df    = pd.DataFrame(X, index=df.index, columns=feats)
X_train = X_df.loc[_idx_tr].values
X_test  = X_df.loc[_idx_te].values
y_train = df.loc[_idx_tr, TARGET_COL].values
y_test  = df.loc[_idx_te, TARGET_COL].values
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_df.shape[1]}")"""
set_src(nb["cells"][10], NEW_10)

save_nb(nb, SALE_PATH)
print("✓ 53_boost_sale_optuna.ipynb modificado")
print(f"  - ratio_dormitorios_superficie eliminado de BASE_FEATURES")
print(f"  - ratio_banos_superficie eliminado de BASE_FEATURES")
print(f"  - precio_m2_municipio_media recalculada desde train (cell 7)")
print(f"  - split adaptado a índices precalculados (cell 10)")

# ═════════════════════════════════════════════════════════════════════════════
# 53_boost_rent.ipynb
# ═════════════════════════════════════════════════════════════════════════════
RENT_PATH = ROOT / "notebooks/05_ML/53_boost_rent.ipynb"
nb = load_nb(RENT_PATH)

# ── Fix 1 + Fix 2 en Cell 10 (BASE_FEATURES + build_X + llamada final) ────────
assert nb["cells"][10]["cell_type"] == "code"
old10r = src(nb["cells"][10])

# a) Quitar ratio features
new10r = old10r.replace(
    '    "ratio_dormitorios_superficie",\n    "ratio_banos_superficie",\n',
    ""
)

# b) Corregir comentario erróneo sobre precio_m2_municipio_media
new10r = new10r.replace(
    '"precio_m2_municipio_media",      # precio medio de VENTA por municipio — no deriva del target',
    '"precio_m2_municipio_media",      # media de precio VENTA por municipio — externa al target de alquiler'
)

# c) Reemplazar bloque final (X, feats = build_X / print)
OLD_END_10R = (
    "X, feats = build_X(df)\n"
    'print(f"Features ({len(feats)}) — sin ninguna derivada del precio:")\n'
    "print(feats)"
)
NEW_END_10R = """\
# ── Fix 2: sustituir precio_m2_municipio_media de alquiler por media de VENTA ──
# Los valores del gold file son medias de alquiler (leakage del target).
# Usamos el precio medio de VENTA por municipio: es externo al target de alquiler,
# captura calidad de zona y no introduce ningún tipo de leakage.
_df_sale = pd.read_csv(
    PROJECT_ROOT / "data" / "gold" / "final_sale_idealistaAPI.csv",
    usecols=lambda c: c.startswith("municipio_") or c == "precio_m2",
)
_sale_mun_oh = [c for c in _df_sale.columns if c.startswith("municipio_")]
_df_sale["_mun"] = _df_sale[_sale_mun_oh].idxmax(axis=1).str.replace("municipio_", "")
_sale_mun_means = _df_sale.groupby("_mun")["precio_m2"].mean()
_sale_global    = float(_df_sale["precio_m2"].mean())

_rent_mun_oh = [c for c in df.columns if c.startswith("municipio_")]
df["_mun"] = df[_rent_mun_oh].idxmax(axis=1).str.replace("municipio_", "")
df["precio_m2_municipio_media"] = df["_mun"].map(_sale_mun_means).fillna(_sale_global)
df = df.drop(columns=["_mun"])

X, feats = build_X(df)
print(f"Features ({len(feats)}) — sin ninguna derivada del precio de alquiler:")
print(feats)"""

new10r = new10r.replace(OLD_END_10R, NEW_END_10R)
assert new10r != old10r, "Cell 10 de rent no cambió — revisar marcadores"
set_src(nb["cells"][10], new10r)

save_nb(nb, RENT_PATH)
print()
print("✓ 53_boost_rent.ipynb modificado")
print(f"  - ratio_dormitorios_superficie eliminado de BASE_FEATURES")
print(f"  - ratio_banos_superficie eliminado de BASE_FEATURES")
print(f"  - precio_m2_municipio_media reemplazada por media de VENTA (cell 10)")
print(f"  - comentario corregido ('VENTA' en lugar de incorrecto 'VENTA sin leakage')")
