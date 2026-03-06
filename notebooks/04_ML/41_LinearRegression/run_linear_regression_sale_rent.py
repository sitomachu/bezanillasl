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
KFOLDS = 5

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
BOOLEAN_FEATURES = ["tiene_garaje", "obra_nueva"]

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


def normalize_boolean_series(series: pd.Series) -> pd.Series:
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


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = pd.Series(True, index=df.index)
    records = []

    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            records.append(
                {
                    "column": col,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower": np.nan,
                    "upper": np.nan,
                    "outliers_column": 0,
                }
            )
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        valid_mask = series.between(lower, upper) | series.isna()

        records.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
                "outliers_column": int((~valid_mask).sum()),
            }
        )

        mask &= valid_mask

    cleaned_df = df.loc[mask].copy()
    stats_df = pd.DataFrame(records)
    stats_df["rows_removed_total"] = int((~mask).sum())

    return cleaned_df, stats_df


def compute_test_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_mean = float(np.mean(y_true))
    nmae = mae / y_mean if y_mean != 0 else np.nan
    nrmse = rmse / y_mean if y_mean != 0 else np.nan

    return {
        "MAE": float(mae),
        "MAPE": float(mape),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "NMAE": float(nmae),
        "NRMSE": float(nrmse),
    }


def run_kfold_metrics(X_train: pd.DataFrame, y_train: pd.Series, use_log_target: bool) -> Dict[str, float]:
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_idx, (idx_tr, idx_val) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[idx_tr].copy()
        X_val = X_train.iloc[idx_val].copy()
        y_tr = y_train.iloc[idx_tr].copy()
        y_val = y_train.iloc[idx_val].copy()

        median_values = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(median_values)
        X_val = X_val.fillna(median_values)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        y_tr_model = np.log(y_tr) if use_log_target else y_tr

        model = LinearRegression()
        model.fit(X_tr_scaled, y_tr_model)

        pred_val_model = model.predict(X_val_scaled)
        pred_val_price = np.exp(pred_val_model) if use_log_target else pred_val_model

        fold_metrics = compute_test_metrics(y_val, pred_val_price)
        fold_metrics["fold"] = fold_idx
        rows.append(fold_metrics)

    cv_df = pd.DataFrame(rows)

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

    vif_values = [
        variance_inflation_factor(X_scaled_df.values, i)
        for i in range(X_scaled_df.shape[1])
    ]

    return pd.DataFrame(
        {
            "variable": X_scaled_df.columns,
            "VIF": vif_values,
        }
    )


def save_residual_plots(y_true: pd.Series, y_pred: np.ndarray, residuals: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(14, 5))

    # Plot 1: y_true vs y_pred
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7, color="steelblue")
    min_axis = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_axis = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_axis, max_axis], [min_axis, max_axis], "r--", linewidth=1.2)
    plt.xlabel("Valor real")
    plt.ylabel("Prediccion")
    plt.title("Real vs Prediccion")

    # Plot 2: histogram of residuals with normal fit
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=25, density=True, alpha=0.6, color="lightblue", edgecolor="black")
    mu, std = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    plt.xlabel("Residuo")
    plt.ylabel("Densidad")
    plt.title("Distribucion de residuos")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def model_specs() -> List[Tuple[int, str, List[str], bool]]:
    specs = []

    case_1 = BASE_FEATURES.copy()
    case_2 = BASE_FEATURES + ["es_unifamiliar"]
    case_3 = BASE_FEATURES + ["es_unifamiliar", "tiene_garaje_bin", "obra_nueva_bin"]
    case_4 = BASE_FEATURES + ["es_unifamiliar"] + LATLON_FEATURES
    case_5 = BASE_FEATURES + ["es_unifamiliar"] + DISTANCE_FEATURES
    case_6 = BASE_FEATURES + ["es_unifamiliar", "precio_m2_municipio"]

    cases = [
        ("Base", case_1),
        ("Base + tipologia", case_2),
        ("Base + tipologia + garaje + obra_nueva", case_3),
        ("Base + tipologia + latitud + longitud", case_4),
        ("Base + tipologia + distancias minimas", case_5),
        ("Base + tipologia + precio_m2_municipio", case_6),
    ]

    model_number = 1
    for case_name, features in cases:
        specs.append((model_number, case_name, features, False))
        model_number += 1
        specs.append((model_number, case_name, features, True))
        model_number += 1

    return specs


def prepare_dataframe_for_modeling(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed_columns = (
        [TARGET_COL, "tipologia", "municipio"]
        + BASE_FEATURES
        + DISTANCE_FEATURES
        + LATLON_FEATURES
        + BOOLEAN_FEATURES
    )

    df = ensure_columns(df_raw, needed_columns).copy()

    # Numeric conversions
    numeric_cols = [TARGET_COL] + BASE_FEATURES + DISTANCE_FEATURES + LATLON_FEATURES
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep valid target rows before outlier filtering
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL] > 0].copy()

    # Outlier removal on key numeric columns
    outlier_cols = [TARGET_COL] + BASE_FEATURES + DISTANCE_FEATURES + LATLON_FEATURES
    outlier_cols = [col for col in outlier_cols if col in df.columns]
    df_clean, outlier_stats = remove_outliers_iqr(df, outlier_cols)

    # Ensure essential columns are not null after cleaning
    for col in BASE_FEATURES:
        df_clean = df_clean[df_clean[col].notna()].copy()

    # Boolean engineering
    tipologia_series = (
        df_clean["tipologia"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df_clean["es_unifamiliar"] = tipologia_series.isin(UNIFAMILIAR_VALUES).astype(int)

    df_clean["tiene_garaje_bin"] = normalize_boolean_series(df_clean["tiene_garaje"]).fillna(0).astype(int)
    df_clean["obra_nueva_bin"] = normalize_boolean_series(df_clean["obra_nueva"]).fillna(0).astype(int)

    # Municipality price-per-m2 feature
    df_clean["precio_m2"] = np.where(
        df_clean["superficie_construida_m2"] > 0,
        df_clean[TARGET_COL] / df_clean["superficie_construida_m2"],
        np.nan,
    )

    precio_m2_municipio = df_clean.groupby("municipio")["precio_m2"].median()
    df_clean["precio_m2_municipio"] = df_clean["municipio"].map(precio_m2_municipio)

    # Fill missing optional numeric columns with median for modeling continuity
    fill_cols = DISTANCE_FEATURES + LATLON_FEATURES + ["precio_m2_municipio"]
    for col in fill_cols:
        median_value = pd.to_numeric(df_clean[col], errors="coerce").median()
        if pd.isna(median_value):
            median_value = 0.0
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(median_value)

    return df_clean, outlier_stats


def fit_single_model(
    df_model: pd.DataFrame,
    dataset_name: str,
    model_num: int,
    case_name: str,
    feature_cols: List[str],
    use_log_target: bool,
    output_base_dir: Path,
) -> Dict[str, float]:
    feature_cols = [col for col in feature_cols if col in df_model.columns]

    X = df_model[feature_cols].copy()
    y = df_model[TARGET_COL].copy()

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

    y_pred_test_model = model.predict(X_test_scaled)
    y_pred_test = np.exp(y_pred_test_model) if use_log_target else y_pred_test_model

    residuals = (y_test - y_pred_test).to_numpy()

    test_metrics = compute_test_metrics(y_test, y_pred_test)
    cv_metrics = run_kfold_metrics(X_train, y_train, use_log_target=use_log_target)

    # Statsmodels for coefficient table and information criteria
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols)
    X_train_sm = sm.add_constant(X_train_scaled_df, has_constant="add")
    sm_results = sm.OLS(y_train_model, X_train_sm).fit()

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
            "y_pred": y_pred_test,
            "residual": residuals,
        }
    )

    model_label = f"M{model_num:02d}_{'log' if use_log_target else 'raw'}"
    model_out_dir = output_base_dir / dataset_name / model_label
    model_out_dir.mkdir(parents=True, exist_ok=True)

    coef_df.to_csv(model_out_dir / "coefficients.csv", index=False)
    vif_df.to_csv(model_out_dir / "vif.csv", index=False)
    residuals_df.to_csv(model_out_dir / "residuals.csv", index=False)

    save_residual_plots(
        y_true=y_test,
        y_pred=y_pred_test,
        residuals=residuals,
        save_path=model_out_dir / "residual_plots.png",
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

    print("\n" + "=" * 110)
    print(f"DATASET: {dataset_name} | MODELO: M{model_num} | TARGET: {'log(precio)' if use_log_target else 'precio'}")
    print(f"Caso: {case_name}")
    print(f"Regresores ({len(feature_cols)}): {feature_cols}")
    print("Metricas test:")
    print({k: round(v, 6) for k, v in test_metrics.items()})
    print("Metricas CV (5-fold):")
    print({k: round(v, 6) for k, v in cv_metrics.items()})
    print(f"AIC: {sm_results.aic:.4f} | BIC: {sm_results.bic:.4f}")
    print(f"Skewness residuos: {metrics_row['residual_skewness']:.6f} | Kurtosis residuos: {metrics_row['residual_kurtosis']:.6f}")
    print("\nCoeficientes (statsmodels):")
    print(coef_df.to_string(index=False))
    print("\nVIF por variable:")
    print(vif_df.to_string(index=False))
    print(f"\nArchivos del modelo guardados en: {model_out_dir}")

    return metrics_row


def run_for_dataset(dataset_name: str, csv_path: Path, output_base_dir: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    print("\n" + "#" * 120)
    print(f"Procesando dataset: {dataset_name} | archivo: {csv_path.name}")
    print(f"Shape original: {df_raw.shape}")

    df_model, outlier_stats = prepare_dataframe_for_modeling(df_raw)

    dataset_out_dir = output_base_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    outlier_stats.to_csv(dataset_out_dir / "outlier_stats.csv", index=False)

    print(f"Shape despues de limpiar outliers y preparar datos: {df_model.shape}")
    print("Resumen de outliers:")
    print(outlier_stats.to_string(index=False))

    rows = []
    for model_num, case_name, feature_cols, use_log_target in model_specs():
        row = fit_single_model(
            df_model=df_model,
            dataset_name=dataset_name,
            model_num=model_num,
            case_name=case_name,
            feature_cols=feature_cols,
            use_log_target=use_log_target,
            output_base_dir=output_base_dir,
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model_num").reset_index(drop=True)

    # Scale in percentage terms for easy reading
    summary["MAPE_pct"] = summary["MAPE"] * 100
    summary["NMAE_pct"] = summary["NMAE"] * 100
    summary["NRMSE_pct"] = summary["NRMSE"] * 100
    summary["CV_MAPE_pct"] = summary["CV_MAPE"] * 100

    display_cols = [
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

    summary_display = summary[display_cols].copy()
    round_cols = [
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
    summary_display[round_cols] = summary_display[round_cols].round(6)

    summary_display.to_csv(dataset_out_dir / "summary_models.csv", index=False)

    print("\nResumen final ordenado por numero de modelo:")
    print(summary_display.to_string(index=False))
    print(f"\nResumen guardado en: {dataset_out_dir / 'summary_models.csv'}")

    return summary_display


def main() -> None:
    project_root = find_project_root(Path.cwd().resolve())
    data_dir = project_root / "data" / "processed" / "idealistaAPI"
    output_dir = project_root / "notebooks" / "04_ML" / "41_LinearRegression" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for dataset_name, filename in DATASET_FILES.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontro el archivo para {dataset_name}: {csv_path}")

        summary_df = run_for_dataset(dataset_name, csv_path, output_base_dir=output_dir)
        all_summaries.append(summary_df)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(output_dir / "summary_all_datasets.csv", index=False)

    print("\n" + "#" * 120)
    print("Resumen combinado (sale + rent) guardado en:")
    print(output_dir / "summary_all_datasets.csv")


if __name__ == "__main__":
    main()
