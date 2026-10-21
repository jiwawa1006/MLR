# plot_figure.py

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split

from feature_selector import (
    feature_count_experiment,
    lasso_experiment,
    elastic_net_experiment
)


# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "fig"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 5

MAX_FEATURES = None
SKIP_EXISTING = True

METHODS = ["variance_threshold", "mutual_information", "forward", "backward", "rfe", "lasso", "elastic_net"]

TARGET_COLUMNS = {
    "amse_housing": "SalePrice",
    "automoblie": "price",
    "boston_housing": "MEDV",
    "cancer_deathrate": "TARGET_deathRate",
    "cover_type": "Cover_Type",
    "e-commerce_customer_behavior": "Total Spend",
    "startups_50": "Profit",
    "student_performance": "G3",
    "wine_quality": "quality",
}

# Utility functions
def clean_index_columns(df):
    """
    Remove CSV index and row ID columns.
    """

    index_columns = [
        col for col in df.columns 
        if col.lower().startswith("unnamed:") or col.lower() in ["id", "customer id", "customer_id", "row_id", "index"]
    ]

    if index_columns:
        print("Removing CSV index/ID columns:")

        for col in index_columns:
            print(f" - {col}")

        df = df.drop(columns=index_columns)

    return df

def detect_target(df):
    """
    Detect target column.

    """

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        raise ValueError("Could not automatically detect target column.")

    return numeric_columns[-1]

def print_selected_features(feature_string):
    """
    Convert a comma-separated feature string into a clean list.
    """

    if pd.isna(feature_string):
        return []

    if isinstance(feature_string, list):
        return feature_string

    feature_string = str(feature_string)
    features = [feature.strip() for feature in feature_string.split(",") if feature.strip()]

    return features

def get_best_rmse_result(results):
    """
    Find the row with the lowest test RMSE.
    """

    if "test_rmse" not in results.columns:
        return None

    return results.loc[results["test_rmse"].idxmin()]


def get_best_r2_result(results):
    """
    Find the row with the highest test R².
    """

    if "test_r2" not in results.columns:
        return None

    return results.loc[results["test_r2"].idxmax()]

# Plotting
def plot_rmse(results, method, output_path):
    """
    Plot Train/Test RMSE against number of features with adaptive clean ticks.
    """
    import matplotlib.ticker as ticker

    plt.figure(figsize=(9, 5.5))
    n_pts = len(results["n_features"])
    marker_style = "." if n_pts <= 20 else None

    plt.plot(results["n_features"], results["train_rmse"], marker=marker_style, linewidth=2, linestyle="--", color="#64748b", label="Train RMSE")
    plt.plot(results["n_features"], results["test_rmse"], marker=marker_style, linewidth=2.2, color="#4f46e5", label="Test RMSE")

    best = get_best_rmse_result(results)

    if best is not None:
        best_k = int(best["n_features"])
        best_val = best["test_rmse"]
        plt.axvline(best_k, color="#dc2626", linestyle=":", alpha=0.6, linewidth=1.5)
        plt.scatter(best_k, best_val, c="#dc2626", s=140, marker="*", zorder=5, label=f"Best Test RMSE (k={best_k}, RMSE={best_val:,.2f})")

    plt.xlabel("Number of Features (k)", fontweight="bold")
    plt.ylabel("RMSE", fontweight="bold")
    plt.title(f"{method.replace('_', ' ').title()} - RMSE vs Feature Count", fontsize=13, fontweight="bold")

    if n_pts <= 20:
        plt.xticks(results["n_features"])
    else:
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12, integer=True))

    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_r2(results, method, output_path):
    """
    Plot Train/Test R² against number of features with adaptive clean ticks.
    """
    import matplotlib.ticker as ticker

    plt.figure(figsize=(9, 5.5))
    n_pts = len(results["n_features"])
    marker_style = "." if n_pts <= 20 else None

    plt.plot(results["n_features"], results["train_r2"], marker=marker_style, linewidth=2, linestyle="--", color="#64748b", label="Train R²")
    plt.plot(results["n_features"], results["test_r2"], marker=marker_style, linewidth=2.2, color="#059669", label="Test R²")

    best = get_best_r2_result(results)

    if best is not None:
        best_k = int(best["n_features"])
        best_val = best["test_r2"]
        plt.axvline(best_k, color="#dc2626", linestyle=":", alpha=0.6, linewidth=1.5)
        plt.scatter(best_k, best_val, c="#dc2626", s=140, marker="*", zorder=5, label=f"Best Test R² (k={best_k}, R²={best_val:.4f})")

    plt.xlabel("Number of Features (k)", fontweight="bold")
    plt.ylabel("R²", fontweight="bold")
    plt.title(f"{method.replace('_', ' ').title()} - R² vs Feature Count", fontsize=13, fontweight="bold")

    if n_pts <= 20:
        plt.xticks(results["n_features"])
    else:
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12, integer=True))

    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Result printing
def print_best_result(results, method):
    """
    Print the best RMSE and R² results.

    Correctly handles selected_features as a
    comma-separated string.
    """

    best_rmse = get_best_rmse_result(results)

    if best_rmse is not None:

        features = print_selected_features(best_rmse["selected_features"])
        
        print(f"\n{method} - Best Test RMSE")
        print(f"k = {int(best_rmse['n_features'])}")
        print(f"RMSE = {best_rmse['test_rmse']:.6f}")
        print(f"Selected features for best RMSE model: {features}")

    best_r2 = get_best_r2_result(results)

    if best_r2 is not None:

        features = print_selected_features(best_r2["selected_features"])
        
        print(f"\n{method} - Best Test R^2")
        print(f"k = {int(best_r2['n_features'])}")
        print(f"R^2 = {best_r2['test_r2']:.6f}")
        print(f"Selected features for best R^2 model: {features}")

# Run one method
def run_method(method, X_train, X_test, y_train, y_test, output_dir, force=False):

    csv_path = os.path.join(output_dir, f"{method}.csv")
    rmse_path = os.path.join(output_dir, f"{method}.png")
    r2_path = os.path.join(output_dir, f"{method}_r2.png")

    if not force and SKIP_EXISTING and os.path.exists(csv_path) and os.path.exists(rmse_path) and os.path.exists(r2_path):
        print(f"Skipping {method} (already computed in {output_dir}).")
        results = pd.read_csv(csv_path)
        print_best_result(results, method)
        return results

    print("\n------------------------------------------------------------")
    print(f"Running {method}")
    print("------------------------------------------------------------")

    if method in ["variance_threshold", "mutual_information", "forward", "backward", "rfe"]:
        results = feature_count_experiment(X_train, X_test, y_train, y_test, method=method, max_features=MAX_FEATURES, cv=CV, random_state=RANDOM_STATE)

    elif method == "lasso":
        results = lasso_experiment(X_train, X_test, y_train, y_test, max_features=MAX_FEATURES, cv=CV, random_state=RANDOM_STATE)

    elif method == "elastic_net":
        results = elastic_net_experiment(X_train, X_test, y_train, y_test, max_features=MAX_FEATURES, l1_ratio=0.5, cv=CV, random_state=RANDOM_STATE)

    else:
        raise ValueError(f"Unknown method: {method}")

    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_rmse(results, method, rmse_path)
    plot_r2(results, method, r2_path)

    print_best_result(results, method)

    return results

# Main
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Multiple Linear Regression feature selection experiments.")
    parser.add_argument("--force", action="store_true", help="Force recomputing experiments even if results exist.")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset name to run (without .csv).")
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{DATA_DIR}'.")
        return

    if args.dataset:
        csv_files = [f for f in csv_files if os.path.splitext(os.path.basename(f))[0].lower() == args.dataset.lower()]
        if not csv_files:
            print(f"Dataset '{args.dataset}' not found in '{DATA_DIR}'.")
            return

    print(f"Found {len(csv_files)} dataset(s) to process.")

    for csv_file in csv_files:

        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

        print("\n======================================================================")
        print(f"Dataset: {dataset_name}")
        print("======================================================================")

        df = pd.read_csv(csv_file)
        df = clean_index_columns(df)
        target = TARGET_COLUMNS.get(dataset_name) or detect_target(df)
        print(f"Target: {target}")

        if df[target].isnull().any():
            missing_count = int(df[target].isnull().sum())
            print(f"Notice: Dropping {missing_count} row(s) with missing target '{target}'.")
            df = df.dropna(subset=[target])

        X = df.drop(columns=[target])
        y = df[target]

        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        if categorical_features:
            print("Categorical columns:")
        
            for col in categorical_features:
                print(f" - {col}")

        print(f"Samples: {len(df)}")
        print(f"Original predictors: {X.shape[1]}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        for method in METHODS:
            run_method(method, X_train, X_test, y_train, y_test, output_dir, force=args.force)

    print("\n======================================================================")
    print("All datasets completed.")
    print("======================================================================")

    try:
        import export_web_data
        print("\nExporting web dashboard data to model_data.js...")
        export_web_data.main()
    except Exception as e:
        print(f"Notice: Could not auto-export web data: {e}")


if __name__ == "__main__":
    main()