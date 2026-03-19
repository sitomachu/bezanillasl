
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SIGMA_THRESHOLD = 3.0

TARGET_COL = "precio"

BASE_FEATURES = [
    "superficie_construida_m2",
    "numero_dormitorios",
    "numero_banos",
]

DISTANCE_FEATURES = [
    "distancia_min_playa_km",
    "distancia_min_supermercado_km",
    "distancia_min_colegio_km",
]

LATLON_FEATURES = ["latitud", "longitud"]
OPTIONAL_BOOL_FEATURES = ["tiene_garaje", "obra_nueva"]

UNIFAMILIAR_VALUES = {
    "chalet",
    "countryhouse",
    "singlefamily",
    "house",
    "villa",
    "townhouse",
    "detachedhouse",
    "adosado",
}

DATASET_FILES = {
    "sale": "sale_homes_clean.csv",
    "rent": "rent_homes_clean.csv",
}


def find_project_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data" / "processed" / "idealistaAPI").exists():
            return candidate
    raise FileNotFoundError("No se encontro la raiz del proyecto con data/processed/idealistaAPI")


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def dedupe_keep_order(columns: List[str]) -> List[str]:
    seen = set()
    result = []
    for col in columns:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def normalize_bool_series(series: pd.Series) -> pd.Series:
    true_values = {"true", "t", "1", "yes", "y", "si", "s"}
    false_values = {"false", "f", "0", "no", "n"}

    def _convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
            return int(value)

        text = str(value).strip().lower()
        if text in true_values:
            return 1
        if text in false_values:
            return 0
        return np.nan

    return series.map(_convert).astype(float)


def remove_outliers_3sigma(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep_mask = pd.Series(True, index=df.index)
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        mean_value = s.mean()
        std_value = s.std(ddof=0)

        if pd.isna(std_value) or std_value == 0:
            col_outliers = pd.Series(False, index=df.index)
            lower = np.nan
            upper = np.nan
        else:
            z = (s - mean_value) / std_value
            col_outliers = z.abs() > SIGMA_THRESHOLD
            lower = mean_value - SIGMA_THRESHOLD * std_value
            upper = mean_value + SIGMA_THRESHOLD * std_value

        keep_mask &= ~col_outliers.fillna(False)

        rows.append(
            {
                "column": col,
                "mean": mean_value,
                "std": std_value,
                "lower_3sigma": lower,
                "upper_3sigma": upper,
                "outliers_column": int(col_outliers.sum()),
            }
        )

    outlier_stats = pd.DataFrame(rows)
    outlier_stats["rows_removed_total"] = int((~keep_mask).sum())

    removed_rows = df.loc[~keep_mask].copy()
    df_clean = df.loc[keep_mask].copy()

    return df_clean, outlier_stats, removed_rows


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_mean = float(np.mean(y_true))
    nmae = float(mae / y_mean) if y_mean != 0 else np.nan
    nrmse = float(rmse / y_mean) if y_mean != 0 else np.nan

    return {
        "MAE": float(mae),
        "MAPE": float(mape),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
        "NMAE": nmae,
        "NRMSE": nrmse,
    }


def compute_cv_metrics(X_train: pd.DataFrame, y_train: pd.Series, use_log_target: bool) -> Dict[str, float]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_id, (idx_tr, idx_val) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[idx_tr].copy()
        X_val = X_train.iloc[idx_val].copy()
        y_tr = y_train.iloc[idx_tr].copy()
        y_val = y_train.iloc[idx_val].copy()

        train_medians = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(train_medians)
        X_val = X_val.fillna(train_medians)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        y_tr_model = np.log(y_tr) if use_log_target else y_tr

        model = LinearRegression()
        model.fit(X_tr_scaled, y_tr_model)

        pred_val_model = model.predict(X_val_scaled)
        pred_val_price = np.exp(pred_val_model) if use_log_target else pred_val_model

        fold_metrics = compute_metrics(y_val, pred_val_price)
        fold_metrics["fold"] = fold_id
        fold_rows.append(fold_metrics)

    cv_df = pd.DataFrame(fold_rows)

    return {
        "CV_MAE": float(cv_df["MAE"].mean()),
        "CV_MAPE": float(cv_df["MAPE"].mean()),
        "CV_MSE": float(cv_df["MSE"].mean()),
        "CV_RMSE": float(cv_df["RMSE"].mean()),
        "CV_R2": float(cv_df["R2"].mean()),
    }


def compute_vif(X_scaled_df: pd.DataFrame) -> pd.DataFrame:
    if X_scaled_df.shape[1] == 0:
        return pd.DataFrame(columns=["variable", "VIF"])

    vifs = [
        variance_inflation_factor(X_scaled_df.values, i)
        for i in range(X_scaled_df.shape[1])
    ]

    return pd.DataFrame({"variable": X_scaled_df.columns, "VIF": vifs})


def save_residual_plots(y_true: pd.Series, y_pred: np.ndarray, residuals: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7, color="steelblue")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], "r--", linewidth=1.2)
    plt.xlabel("Valor real")
    plt.ylabel("Prediccion")
    plt.title("Real vs Prediccion")

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=25, density=True, alpha=0.65, color="lightblue", edgecolor="black")
    mu, std = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 300)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    plt.xlabel("Residuo")
    plt.ylabel("Densidad")
    plt.title("Distribucion de residuos")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_visual_table(df_summary: pd.DataFrame, title: str, output_html: Path) -> None:
    show_cols = [
        "dataset",
        "model_num",
        "model_name",
        "case_name",
        "target",
        "features",
        "n_features",
        "MAE",
        "MAPE_pct",
        "MSE",
        "RMSE",
        "R2",
        "CV_MAE",
        "CV_MAPE_pct",
        "CV_MSE",
        "CV_RMSE",
        "CV_R2",
        "AIC",
        "BIC",
        "residual_skewness",
        "residual_kurtosis",
        "NMAE_pct",
        "NRMSE_pct",
    ]

    visual_df = df_summary[show_cols].copy()

    fmt = {
        "MAE": "{:.2f}",
        "MAPE_pct": "{:.2f}",
        "MSE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R2": "{:.4f}",
        "CV_MAE": "{:.2f}",
        "CV_MAPE_pct": "{:.2f}",
        "CV_MSE": "{:.2f}",
        "CV_RMSE": "{:.2f}",
        "CV_R2": "{:.4f}",
        "AIC": "{:.2f}",
        "BIC": "{:.2f}",
        "residual_skewness": "{:.4f}",
        "residual_kurtosis": "{:.4f}",
        "NMAE_pct": "{:.2f}",
        "NRMSE_pct": "{:.2f}",
    }

    styler = (
        visual_df.style
        .set_caption(title)
        .format(fmt)
        .background_gradient(subset=["R2", "CV_R2"], cmap="RdYlGn")
        .background_gradient(subset=["MAE", "RMSE", "CV_MAE", "CV_RMSE", "NMAE_pct", "NRMSE_pct"], cmap="YlGnBu_r")
        .background_gradient(subset=["AIC", "BIC"], cmap="PuBu_r")
        .set_properties(**{"text-align": "center", "font-size": "11px"})
        .set_properties(subset=["features"], **{"text-align": "left"})
    )

    html = styler.to_html()
    output_html.write_text(html, encoding="utf-8")


def build_case_definitions() -> List[Tuple[str, List[str]]]:
    base = BASE_FEATURES.copy()
    tipologia = BASE_FEATURES + ["dummy_tipologia_unifamiliar"]
    tipologia_garaje_obra = tipologia + ["tiene_garaje_bin", "obra_nueva_bin"]
    tipologia_geo = tipologia + LATLON_FEATURES
    tipologia_dist = tipologia + DISTANCE_FEATURES
    all_features = tipologia_garaje_obra + LATLON_FEATURES + DISTANCE_FEATURES

    cases = [
        ("Caso 1: Base", base),
        ("Caso 1b: Base + precio_m2_municipio", base + ["precio_m2_municipio"]),
        ("Caso 2: Base + dummy tipologia", tipologia),
        ("Caso 2b: Caso 2 + precio_m2_municipio", tipologia + ["precio_m2_municipio"]),
        ("Caso 3: Caso 2 + garaje + obra_nueva", tipologia_garaje_obra),
        ("Caso 3b: Caso 3 + precio_m2_municipio", tipologia_garaje_obra + ["precio_m2_municipio"]),
        ("Caso 4: Caso 2 + latitud + longitud", tipologia_geo),
        ("Caso 4b: Caso 4 + precio_m2_municipio", tipologia_geo + ["precio_m2_municipio"]),
        ("Caso 5: Caso 2 + distancias minimas", tipologia_dist),
        ("Caso 5b: Caso 5 + precio_m2_municipio", tipologia_dist + ["precio_m2_municipio"]),
        ("Caso 6: Todas las variables", all_features),
        ("Caso 6b: Todas las variables + precio_m2_municipio", all_features + ["precio_m2_municipio"]),
    ]

    return [(name, dedupe_keep_order(features)) for name, features in cases]


def build_model_specs() -> List[Tuple[int, str, List[str], bool]]:
    specs = []
    model_num = 1
    for case_name, features in build_case_definitions():
        specs.append((model_num, case_name, features, False))
        model_num += 1
        specs.append((model_num, case_name, features, True))
        model_num += 1
    return specs


def prepare_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    needed = [
        TARGET_COL,
        "tipologia",
        "municipio",
    ] + BASE_FEATURES + DISTANCE_FEATURES + LATLON_FEATURES + OPTIONAL_BOOL_FEATURES

    df = ensure_columns(df_raw, needed).copy()

    # Ensure target valid
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL] > 0].copy()

    # Numeric conversion before outlier detection
    numeric_for_outliers = [TARGET_COL] + BASE_FEATURES + DISTANCE_FEATURES + LATLON_FEATURES
    for col in numeric_for_outliers:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3-sigma outlier removal
    df_clean, outlier_stats, removed_rows = remove_outliers_3sigma(df, numeric_for_outliers)

    # Fill numerics for model continuity
    model_numeric_fill = BASE_FEATURES + DISTANCE_FEATURES + LATLON_FEATURES
    for col in model_numeric_fill:
        median_value = pd.to_numeric(df_clean[col], errors="coerce").median()
        if pd.isna(median_value):
            median_value = 0.0
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(median_value)

    # One dummy for apartment/unifamiliar (1 unifamiliar, 0 apartment/other)
    tipologia_text = df_clean["tipologia"].fillna("").astype(str).str.strip().str.lower()
    df_clean["dummy_tipologia_unifamiliar"] = tipologia_text.isin(UNIFAMILIAR_VALUES).astype(int)

    # Additional booleans for the requested model set
    df_clean["tiene_garaje_bin"] = normalize_bool_series(df_clean["tiene_garaje"]).fillna(0).astype(int)
    df_clean["obra_nueva_bin"] = normalize_bool_series(df_clean["obra_nueva"]).fillna(0).astype(int)

    # Price-per-m2 and municipality median mapping
    df_clean["precio_m2"] = np.where(
        df_clean["superficie_construida_m2"] > 0,
        df_clean[TARGET_COL] / df_clean["superficie_construida_m2"],
        np.nan,
    )

    municipio_map = df_clean.groupby("municipio")["precio_m2"].median()
    df_clean["precio_m2_municipio"] = df_clean["municipio"].map(municipio_map)

    median_pm2 = pd.to_numeric(df_clean["precio_m2_municipio"], errors="coerce").median()
    if pd.isna(median_pm2):
        median_pm2 = 0.0
    df_clean["precio_m2_municipio"] = pd.to_numeric(df_clean["precio_m2_municipio"], errors="coerce").fillna(median_pm2)

    return df_clean, outlier_stats, removed_rows


def fit_single_model(
    df_model: pd.DataFrame,
    dataset_name: str,
    model_num: int,
    case_name: str,
    feature_cols: List[str],
    use_log_target: bool,
    output_root: Path,
) -> Dict[str, float]:
    X = df_model[feature_cols].copy()
    y = df_model[TARGET_COL].copy()

    # Same split for all models within each dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_model = np.log(y_train) if use_log_target else y_train

    model = LinearRegression()
    model.fit(X_train_scaled, y_train_model)

    pred_test_model = model.predict(X_test_scaled)
    pred_test_price = np.exp(pred_test_model) if use_log_target else pred_test_model

    residuals = (y_test - pred_test_price).to_numpy()

    test_metrics = compute_metrics(y_test, pred_test_price)
    cv_metrics = compute_cv_metrics(X_train, y_train, use_log_target=use_log_target)

    # statsmodels for detail / aic / bic
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols)
    X_sm = sm.add_constant(X_train_scaled_df, has_constant="add")
    sm_results = sm.OLS(y_train_model, X_sm).fit()

    coef_df = pd.DataFrame(
        {
            "variable": sm_results.params.index,
            "coef": sm_results.params.values,
            "std_error": sm_results.bse.values,
            "t_value": sm_results.tvalues.values,
            "p_value": sm_results.pvalues.values,
        }
    ).round(6)

    vif_df = compute_vif(X_train_scaled_df).round(6)

    residuals_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": pred_test_price,
            "residual": residuals,
        }
    )

    model_label = f"M{model_num:02d}_{'log' if use_log_target else 'raw'}"
    model_dir = output_root / dataset_name / model_label
    model_dir.mkdir(parents=True, exist_ok=True)

    coef_df.to_csv(model_dir / "coefficients.csv", index=False)
    vif_df.to_csv(model_dir / "vif.csv", index=False)
    residuals_df.to_csv(model_dir / "residuals.csv", index=False)

    save_residual_plots(
        y_true=y_test,
        y_pred=pred_test_price,
        residuals=residuals,
        save_path=model_dir / "residual_plots.png",
    )

    metrics_row = {
        "dataset": dataset_name,
        "model_num": model_num,
        "model_name": f"M{model_num}",
        "case_name": case_name,
        "target": "log(precio)" if use_log_target else "precio",
        "features": ", ".join(feature_cols),
        "n_features": len(feature_cols),
        "AIC": float(sm_results.aic),
        "BIC": float(sm_results.bic),
        "residual_skewness": float(skew(residuals, bias=False)),
        "residual_kurtosis": float(kurtosis(residuals, bias=False, fisher=True)),
    }

    metrics_row.update(test_metrics)
    metrics_row.update(cv_metrics)

    print("\n" + "=" * 120)
    print(f"DATASET: {dataset_name} | MODELO: M{model_num} | TARGET: {metrics_row['target']}")
    print(f"Caso: {case_name}")
    print(f"Regresores ({len(feature_cols)}): {feature_cols}")
    print("Metricas test:")
    print({k: round(v, 6) for k, v in test_metrics.items()})
    print("Metricas CV (5-fold):")
    print({k: round(v, 6) for k, v in cv_metrics.items()})
    print(f"AIC: {metrics_row['AIC']:.4f} | BIC: {metrics_row['BIC']:.4f}")
    print(f"Skewness residuos: {metrics_row['residual_skewness']:.6f}")
    print(f"Kurtosis residuos: {metrics_row['residual_kurtosis']:.6f}")
    print("\nCoeficientes:")
    print(coef_df.to_string(index=False))
    print("\nVIF:")
    print(vif_df.to_string(index=False))
    print(f"\nSalida modelo: {model_dir}")

    return metrics_row


def run_dataset(dataset_name: str, csv_path: Path, output_root: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    print("\n" + "#" * 120)
    print(f"Procesando dataset: {dataset_name} | {csv_path.name}")
    print(f"Shape original: {df_raw.shape}")

    df_model, outlier_stats, removed_rows = prepare_dataframe(df_raw)

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    outlier_stats.to_csv(dataset_dir / "outlier_stats_3sigma.csv", index=False)
    removed_rows.to_csv(dataset_dir / "rows_removed_outliers.csv", index=False)

    print(f"Shape despues de outliers 3 sigma: {df_model.shape}")
    print("Resumen outliers (3 sigma):")
    print(outlier_stats.to_string(index=False))

    rows = []
    for model_num, case_name, features, use_log in build_model_specs():
        row = fit_single_model(
            df_model=df_model,
            dataset_name=dataset_name,
            model_num=model_num,
            case_name=case_name,
            feature_cols=features,
            use_log_target=use_log,
            output_root=output_root,
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model_num").reset_index(drop=True)

    summary["MAPE_pct"] = summary["MAPE"] * 100
    summary["NMAE_pct"] = summary["NMAE"] * 100
    summary["NRMSE_pct"] = summary["NRMSE"] * 100
    summary["CV_MAPE_pct"] = summary["CV_MAPE"] * 100

    # Save machine-readable summary
    summary.to_csv(dataset_dir / "summary_models_full.csv", index=False)

    # Save visual summary
    create_visual_table(
        df_summary=summary,
        title=f"Resumen Modelos Lineales - {dataset_name.upper()}",
        output_html=dataset_dir / "summary_models_visual.html",
    )

    # Compact csv for quick reading
    quick_cols = [
        "dataset",
        "model_num",
        "model_name",
        "case_name",
        "target",
        "n_features",
        "MAE",
        "MAPE_pct",
        "MSE",
        "RMSE",
        "R2",
        "AIC",
        "BIC",
        "residual_skewness",
        "residual_kurtosis",
        "CV_MAE",
        "CV_MAPE_pct",
        "CV_MSE",
        "CV_RMSE",
        "CV_R2",
        "NMAE_pct",
        "NRMSE_pct",
    ]

    quick = summary[quick_cols].copy()
    numeric_cols = [
        "MAE",
        "MAPE_pct",
        "MSE",
        "RMSE",
        "R2",
        "AIC",
        "BIC",
        "residual_skewness",
        "residual_kurtosis",
        "CV_MAE",
        "CV_MAPE_pct",
        "CV_MSE",
        "CV_RMSE",
        "CV_R2",
        "NMAE_pct",
        "NRMSE_pct",
    ]
    quick[numeric_cols] = quick[numeric_cols].round(6)
    quick.to_csv(dataset_dir / "summary_models.csv", index=False)

    print("\nResumen final (orden por numero de modelo):")
    print(quick.to_string(index=False))
    print(f"\nResumen CSV: {dataset_dir / 'summary_models.csv'}")
    print(f"Resumen visual HTML: {dataset_dir / 'summary_models_visual.html'}")

    return summary


def explain_negative_r2_sale(summary_sale: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    sale = summary_sale.copy()

    neg = sale[sale["R2"] < 0].copy()

    if neg.empty:
        diagnosis = pd.DataFrame(
            [
                {
                    "diagnosis": "No hay modelos con R2 negativo en SALE tras el nuevo pipeline 3-sigma.",
                    "details": "Aun asi, revisa MAPE y RMSE para detectar errores absolutos altos.",
                }
            ]
        )
    else:
        overall = sale[["model_num", "case_name", "target", "R2", "RMSE", "MAE", "NMAE_pct", "NRMSE_pct"]].copy()
        overall = overall.sort_values("R2")

        diagnosis_rows = []
        for _, row in neg.iterrows():
            reason = (
                "R2 negativo indica que el modelo predice peor que usar la media del precio en test. "
                "En estas especificaciones suele pasar por baja capacidad explicativa del conjunto de regresores "
                "y por heterogeneidad de precios no capturada (ubicacion fina/calidad)."
            )
            diagnosis_rows.append(
                {
                    "model_num": int(row["model_num"]),
                    "target": row["target"],
                    "case_name": row["case_name"],
                    "R2": float(row["R2"]),
                    "MAE": float(row["MAE"]),
                    "RMSE": float(row["RMSE"]),
                    "NMAE_pct": float(row["NMAE_pct"]),
                    "NRMSE_pct": float(row["NRMSE_pct"]),
                    "main_reason": reason,
                }
            )

        diagnosis = pd.DataFrame(diagnosis_rows)
        overall.to_csv(output_root / "sale" / "sale_r2_overview.csv", index=False)

    diagnosis.to_csv(output_root / "sale" / "sale_negative_r2_diagnosis.csv", index=False)
    return diagnosis


def main() -> None:
    project_root = find_project_root(Path.cwd().resolve())
    data_dir = project_root / "data" / "processed" / "idealistaAPI"

    # All outputs in data/ML (as requested)
    output_root = project_root / "data" / "ML" / "linear_regression"
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    summary_by_dataset = {}

    for dataset_name, filename in DATASET_FILES.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontro el archivo: {csv_path}")

        summary = run_dataset(dataset_name, csv_path, output_root)
        all_summaries.append(summary)
        summary_by_dataset[dataset_name] = summary

    combined = pd.concat(all_summaries, ignore_index=True)
    combined = combined.sort_values(["dataset", "model_num"]).reset_index(drop=True)

    combined.to_csv(output_root / "summary_all_models_full.csv", index=False)

    combined_quick = combined.copy()
    combined_quick["MAPE_pct"] = combined_quick["MAPE"] * 100
    combined_quick["NMAE_pct"] = combined_quick["NMAE"] * 100
    combined_quick["NRMSE_pct"] = combined_quick["NRMSE"] * 100
    combined_quick["CV_MAPE_pct"] = combined_quick["CV_MAPE"] * 100
    combined_quick.to_csv(output_root / "summary_all_models.csv", index=False)

    create_visual_table(
        df_summary=combined_quick,
        title="Resumen General Modelos Lineales (SALE + RENT)",
        output_html=output_root / "summary_all_models_visual.html",
    )

    if "sale" in summary_by_dataset:
        sale_summary = summary_by_dataset["sale"].copy()
        sale_summary["NMAE_pct"] = sale_summary["NMAE"] * 100
        sale_summary["NRMSE_pct"] = sale_summary["NRMSE"] * 100
        explain_negative_r2_sale(sale_summary, output_root)

    print("\n" + "#" * 120)
    print("Proceso completado.")
    print(f"Salida principal: {output_root}")
    print(f"Tabla visual general: {output_root / 'summary_all_models_visual.html'}")


if __name__ == "__main__":
    main()
