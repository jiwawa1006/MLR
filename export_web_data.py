# export_web_data.py
"""
Exports comprehensive Multiple Linear Regression model data for all datasets
in `data/` to `model_data.js`, formatted for the interactive GitHub Pages dashboard.
Automatically supports new datasets added to `data/`.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from feature_selector import FeatureSelector

DATA_DIR = "data"
FIG_DIR = "fig"
OUTPUT_JS = "model_data.js"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 5
MAX_FEATURES = 5

TARGET_COLUMNS = {
    "startups_50": "Profit",
    "boston_housing": "MEDV",
    "cancer_deathrate": "TARGET_deathRate",
    "student_performance": "G3",
    "cover_type": "Cover_Type",
    "amse_housing": "SalePrice",
}

DATASET_DISPLAY_NAMES = {
    "amse_housing": "Ames Housing (Sale Price Evaluation)",
    "startups_50": "50 Startups (Profit Prediction)",
    "boston_housing": "Boston Housing (Price Valuation)",
    "cancer_deathrate": "US Cancer Mortality (Death Rate)",
    "student_performance": "Student Performance (Final Grade G3)",
    "cover_type": "Forest Cover Type (Cartographic Prediction)",
}

DATASET_DESCRIPTIONS = {
    "startups_50": "Predict profit for 50 commercial startups based on R&D, Administration, Marketing Spend, and State location. Explores the 2-feature sweet spot vs overfitting.",
    "boston_housing": "Predict median home values (MEDV in $1000s) across 506 Boston suburbs using socio-economic and environmental predictors (RM, LSTAT, PTRATIO, etc.).",
    "cancer_deathrate": "Predict county-level cancer mortality rates (per 100,000 residents) using socio-economic, healthcare coverage, and clinical demographic variables.",
    "student_performance": "Predict secondary student final grades (G3 scale 0-20) based on demographic, social, and academic history predictors.",
    "cover_type": "Predict forest cover classification (7 species categories) from USGS wilderness cartographic, elevation, hydrology, and soil attributes.",
    "amse_housing": "Predict residential home sale prices across Ames, Iowa using 80+ structural, quality, neighborhood, and remodel year predictors.",
}

TARGET_UNITS = {
    "startups_50": "$",
    "boston_housing": "k$",
    "cancer_deathrate": "per 100k",
    "student_performance": "pts",
    "cover_type": "Class",
    "amse_housing": "$",
}

METHODS = [
    "lasso",
    "forward",
    "backward",
    "rfe",
    "elastic_net",
    "mutual_information",
    "variance_threshold",
]

METHOD_DISPLAY_NAMES = {
    "lasso": "Lasso (L1)",
    "forward": "Forward SFS",
    "backward": "Backward Elim.",
    "rfe": "RFE",
    "elastic_net": "ElasticNet",
    "mutual_information": "Mutual Info",
    "variance_threshold": "Variance Thresh",
}

METHOD_BADGES = {
    "lasso": "L1 Sparsity",
    "forward": "Greedy Stepwise",
    "backward": "Elimination",
    "rfe": "Recursive",
    "elastic_net": "L1 + L2",
    "mutual_information": "Nonlinear",
    "variance_threshold": "Unsupervised",
}


def clean_index_columns(df):
    index_columns = [col for col in df.columns if col.lower().startswith("unnamed:")]
    if index_columns:
        df = df.drop(columns=index_columns)
    return df


def detect_target(df):
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("Could not automatically detect target column.")
    return numeric_columns[-1]


def parse_selected_features(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    return [f.strip() for f in str(val).split(",") if f.strip()]


def compute_ols_stats(X_train_sub, y_train, X_test_sub, y_test, feature_names):
    """
    Fits Ordinary Least Squares and extracts coefficients, standard errors,
    t-stats, p-values, 95% confidence intervals, and F-statistic.
    """
    n_train = len(y_train)
    p = len(feature_names)

    # Design matrix with intercept for stats
    X_mat = np.column_stack([np.ones(n_train), X_train_sub.values])
    y_vec = y_train.values

    # Fit scikit-learn model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_sub, y_train)

    train_pred = model.predict(X_train_sub)
    test_pred = model.predict(X_test_sub)

    # Residuals & variance estimates on train
    train_res = y_vec - train_pred
    dof = max(1, n_train - p - 1)
    sigma2 = np.sum(train_res ** 2) / dof

    # Covariance matrix (X^T X)^-1 * sigma^2
    try:
        XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
        se = np.sqrt(np.maximum(0, np.diag(XtX_inv) * sigma2))
    except Exception:
        se = np.zeros(p + 1)

    all_coefs = np.array([model.intercept_] + list(model.coef_))
    all_names = ["Intercept (β₀)"] + feature_names

    t_stats = []
    p_values = []
    ci_lower = []
    ci_upper = []
    t_crit = stats.t.ppf(0.975, dof)

    for beta, stderr in zip(all_coefs, se):
        if stderr > 1e-12:
            t = beta / stderr
            pval = 2 * (1 - stats.t.cdf(abs(t), dof))
        else:
            t = 0.0
            pval = 1.0
        t_stats.append(t)
        p_values.append(pval)
        ci_lower.append(beta - t_crit * stderr)
        ci_upper.append(beta + t_crit * stderr)

    coef_details = []
    for i, name in enumerate(all_names):
        coef_details.append({
            "feature": name,
            "coef": round(float(all_coefs[i]), 4),
            "stderr": round(float(se[i]), 4),
            "tstat": round(float(t_stats[i]), 3),
            "pvalue": round(float(p_values[i]), 5),
            "ci_lower": round(float(ci_lower[i]), 3),
            "ci_upper": round(float(ci_upper[i]), 3),
        })

    # Overall F-statistic
    ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
    ss_res = np.sum(train_res ** 2)
    ss_reg = max(0, ss_tot - ss_res)
    f_stat = (ss_reg / p) / (ss_res / dof) if (p > 0 and ss_res > 1e-12) else 0.0
    f_pval = 1 - stats.f.cdf(f_stat, p, dof) if p > 0 else 1.0

    # Test metrics
    test_rmse = round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4)
    train_rmse = round(float(np.sqrt(mean_squared_error(y_train, train_pred))), 4)
    train_r2 = round(float(r2_score(y_train, train_pred)), 4)
    test_r2 = round(float(r2_score(y_test, test_pred)), 4)
    mae = round(float(mean_absolute_error(y_test, test_pred)), 4)

    n_test = len(y_test)
    adj_r2 = round(float(1 - (1 - test_r2) * (n_test - 1) / max(1, n_test - p - 1)), 4)

    # Test Outliers / Residuals
    residuals = y_test.values - test_pred
    abs_errors = np.abs(residuals)
    outliers = []
    sort_idx = np.argsort(abs_errors)[::-1]

    for rank, idx in enumerate(sort_idx[:10], start=1):
        actual_val = float(y_test.values[idx])
        pred_val = float(test_pred[idx])
        res_val = float(residuals[idx])
        pct_err = float(abs_errors[idx] / max(abs(actual_val), 1e-6) * 100)
        orig_id = int(y_test.index[idx]) if hasattr(y_test.index, "values") else idx

        status = "normal"
        if pct_err > 15:
            status = "alert"
        elif pct_err > 8:
            status = "warning"

        outliers.append({
            "rank": rank,
            "index": orig_id,
            "actual": round(actual_val, 2),
            "predicted": round(pred_val, 2),
            "residual": round(res_val, 2),
            "abs_error": round(float(abs_errors[idx]), 2),
            "pct_error": round(pct_err, 2),
            "status": status,
        })

    return {
        "coefficients": coef_details,
        "f_stat": round(float(f_stat), 2),
        "f_pvalue": round(float(f_pval), 5),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "adj_r2": adj_r2,
        "mae": mae,
        "outliers": outliers,
    }


def process_dataset(csv_path):
    dataset_id = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\nProcessing dataset: {dataset_id}")

    df = pd.read_csv(csv_path)
    df = clean_index_columns(df)

    target_col = TARGET_COLUMNS.get(dataset_id) or detect_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Preprocess
    selector = FeatureSelector(method="lasso", random_state=RANDOM_STATE)
    X_train_proc, X_test_proc = selector._prepare_features(X_train, X_test)
    feature_names = X_train_proc.columns.tolist()

    # Feature metadata for interactive calculator sliders
    feature_stats = {}
    for col in feature_names:
        series = X_train_proc[col]
        min_v = float(series.min())
        max_v = float(series.max())
        mean_v = float(series.mean())
        median_v = float(series.median())
        val_range = max_v - min_v

        if val_range <= 1:
            step = 0.05
        elif val_range <= 10:
            step = 0.1
        elif val_range <= 100:
            step = 1.0
        elif val_range <= 10000:
            step = 100.0
        else:
            step = 1000.0

        feature_stats[col] = {
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "median": median_v,
            "step": step,
            "default": median_v,
        }

    dataset_result = {
        "id": dataset_id,
        "name": DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id.replace("_", " ").title()),
        "description": DATASET_DESCRIPTIONS.get(dataset_id, f"Analysis of {dataset_id} with Multiple Linear Regression."),
        "target": target_col,
        "unit": TARGET_UNITS.get(dataset_id, ""),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_original_features": X.shape[1],
        "n_encoded_features": len(feature_names),
        "features": feature_names,
        "feature_stats": feature_stats,
        "methods": {},
        "sweet_spot": {"k": 2, "method": "lasso", "test_rmse": float("inf")},
    }

    min_overall_rmse = float("inf")
    best_overall_k = 2
    best_overall_method = "lasso"

    for method in METHODS:
        csv_file = os.path.join(FIG_DIR, dataset_id, f"{method}.csv")
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping {method}.")
            continue

        method_df = pd.read_csv(csv_file)
        k_results = []

        for _, row in method_df.iterrows():
            k = int(row["n_features"])
            selected_features = parse_selected_features(row["selected_features"])

            # Filter valid features
            selected_features = [f for f in selected_features if f in feature_names]

            X_tr_sub = X_train_proc[selected_features]
            X_te_sub = X_test_proc[selected_features]

            # OLS and detailed stats
            stats_dict = compute_ols_stats(X_tr_sub, y_train, X_te_sub, y_test, selected_features)

            # Preserve precomputed cv_rmse if present
            cv_rmse = float(row.get("cv_rmse", stats_dict["test_rmse"]))

            feat_map = {name: i for i, name in enumerate(dataset_result["features"])}
            fi_list = [feat_map[f] for f in selected_features if f in feat_map]
            c_list = [[round(c["coef"], 4), round(c["stderr"], 4)] for c in stats_dict["coefficients"]]
            o_list = [[o["index"], round(o["actual"], 2), round(o["predicted"], 2)] for o in stats_dict["outliers"]]

            k_data = {
                "k": k,
                "fi": fi_list,
                "train_rmse": stats_dict["train_rmse"],
                "test_rmse": stats_dict["test_rmse"],
                "train_r2": stats_dict["train_r2"],
                "test_r2": stats_dict["test_r2"],
                "adj_r2": stats_dict["adj_r2"],
                "mae": stats_dict["mae"],
                "cv_rmse": cv_rmse,
                "f_stat": stats_dict["f_stat"],
                "f_pvalue": stats_dict["f_pvalue"],
                "c": c_list,
                "o": o_list,
            }

            if stats_dict["test_rmse"] < min_overall_rmse:
                min_overall_rmse = stats_dict["test_rmse"]
                best_overall_k = k
                best_overall_method = method

            k_results.append(k_data)

        dataset_result["methods"][method] = {
            "key": method,
            "name": METHOD_DISPLAY_NAMES.get(method, method.title()),
            "badge": METHOD_BADGES.get(method, "ML Selection"),
            "data": k_results,
        }

    dataset_result["sweet_spot"] = {
        "k": best_overall_k,
        "method": best_overall_method,
        "test_rmse": min_overall_rmse,
    }

    print(f"Done {dataset_id}: Best overall sweet spot k={best_overall_k} via {best_overall_method} (RMSE: {min_overall_rmse:.2f})")
    return dataset_result


def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}.")
        return

    all_datasets = {}

    for csv_file in csv_files:
        ds_id = os.path.splitext(os.path.basename(csv_file))[0]
        ds_data = process_dataset(csv_file)
        all_datasets[ds_id] = ds_data

    # Write model_data.js
    json_str = json.dumps(all_datasets, separators=(',', ':'))
    js_content = f"// Auto-generated by export_web_data.py\nwindow.DATASETS_DATA = {json_str};\n"

    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"\nSuccessfully generated {OUTPUT_JS} with {len(all_datasets)} dataset(s).")


if __name__ == "__main__":
    main()
