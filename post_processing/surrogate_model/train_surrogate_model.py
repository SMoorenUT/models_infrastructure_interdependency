from __future__ import annotations
from pathlib import Path
from time import time

import os
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import joblib
import argparse
import re
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr
import warnings
import seaborn as sns

"""
surrogate_model.py

Utilities for training, evaluating and saving regression surrogate models from CSV data.
Designed for quick experimentation with multiple models and feature importance analysis.

Key features:
- Load CSV and infer or specify target column(s)
- Configurable output column omission (via NUMBER_OF_OUTPUT_COLUMNS_IN_CSV)
- Automated preprocessing: numeric (impute + scale), categorical (impute + one-hot encode)
- Multiple model support: Random Forest, Extra Trees, Gradient Boosting, Linear, SVM, Gaussian Process, Polynomial
- Feature importance analysis: correlation, mutual information, model-based, permutation
- Correlation analysis with heatmap visualization
- Evaluation metrics: RMSE, R2, MAE, NRMSE, RMSE/std
- Visualization: parity plots, residuals, feature importance comparisons
- Pipeline persistence: save/load with joblib
- Automatic matplotlib backend detection (interactive/headless)

Default configuration:
- DEFAULT_CSV: path to input CSV
- DEFAULT_MODEL_OUT: path to save trained pipeline
- NUMBER_OF_OUTPUT_COLUMNS_IN_CSV: columns to exclude from features
- TARGET_COLUMN: name or index of target variable
- MODEL_NAME: default model type

Main functions:
- load_csv, omit_columns, infer_target: data loading and preparation
- build_preprocessor, make_model, build_pipeline: model construction
- train_test_pipeline, train_from_csv: training workflows
- save_pipeline, load_pipeline, predict_from_df: persistence and inference
- evaluate_feature_importance, analyze_correlations: analysis tools
- train_surrogate_model, train_multiple_surrogate_models: CLI/batch training

CLI usage:
    python surrogate_model.py --csv data.csv --target Y --out model.joblib
    python surrogate_model.py --omit col1,col2

Programmatic usage:
    df = load_csv("data.csv")
    pipeline, metrics = train_test_pipeline(df, target_col="Y", model_name="extra_trees")
    save_pipeline(pipeline, "model.joblib")
    importance = evaluate_feature_importance(df, "Y", pipeline)
"""

CURR_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = CURR_DIR

DEFAULT_CSV = "/home/moorens/code/analysis/ema_road_model_17_07_2025_results.csv"
DEFAULT_MODEL_OUT = OUTPUT_DIR / "pipelines" / "surrogate_pipeline.joblib"

NUMBER_OF_OUTPUT_COLUMNS_IN_CSV = 36  # Example: last 36 columns are outputs
TARGET_COLUMN = "RoadSegment_551_transport.passenger_car_unit_year_2050" # Can be str (name) or int (index); set to None to default to last column

MODEL_NAME = "gradient_boosting"  # Default model type
MODEL_NAMES = ["random_forest", "extra_trees", "gradient_boosting", "linear", "xgboost", "lightgbm", "catboost", "gaussian_process"]
models = MODEL_NAMES
SAVE_PLOTS = True  # Set to True to save plots
SHOW_PLOTS = False  # Set to True to show plots

# Output logging setup
OUTPUT_LOG = OUTPUT_DIR / "surrogate_model_logs" / f"surrogate_model_output_{TARGET_COLUMN.replace('.', '_')}.txt"
_log_file = None

def init_log():
    """Initialize log file for direct output."""
    global _log_file
    _log_file = open(OUTPUT_LOG, 'w')
    print(f"Logging output to: {OUTPUT_LOG}")

def close_log():
    """Close log file."""
    global _log_file
    if _log_file:
        _log_file.close()
        print(f"Log saved to: {OUTPUT_LOG}")

def log_print(*args, **kwargs):
    """Print to both console and log file."""
    # Print to console
    print(*args, **kwargs)
    # Print to file
    if _log_file:
        print(*args, **kwargs, file=_log_file)
        _log_file.flush()  # Ensure immediate write

def load_csv(path: str = DEFAULT_CSV) -> pd.DataFrame:
        """Load CSV into a DataFrame."""
        if not os.path.exists(path):
                raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
                raise ValueError("Loaded CSV is empty")
        return df

def omit_columns(df: pd.DataFrame, target_column: Union[str, int, list, tuple]) -> pd.DataFrame:
        """Omit specified columns from the dataframe.

        `target_column` may be:
        - str: single column name (or comma-separated names)
        - int: index of a column to omit

        Raises TypeError if `target_column` has an invalid type,
        KeyError if any named columns are missing.
        """
        cols_to_omit = df.columns[-NUMBER_OF_OUTPUT_COLUMNS_IN_CSV:]

        # Resolve integer index to a column name
        if isinstance(target_column, int):
                try:
                        target_column = df.columns[target_column]
                except IndexError:
                        raise IndexError(f"target_column index {target_column} out of range for dataframe with {len(df.columns)} columns")

        if target_column is None:
                raise ValueError("target_column must be specified")

        # Normalize to a list of stripped column names
        if isinstance(target_column, str):
                cols_to_omit = cols_to_omit.drop(target_column)
        else:
                raise TypeError("target_column must be a string or integer index")

        return df.drop(columns=cols_to_omit)

def infer_target(df: pd.DataFrame, target_col: Optional[str]) -> str:
        """Return a target column name; default to the last column if None."""
        if target_col:
                if target_col not in df.columns:
                        raise KeyError(f"target_col '{target_col}' not found in dataframe")
                return target_col
        return df.columns[-1]

def analyze_correlations(
    df: pd.DataFrame,
    up_to_column: Optional[str] = None,
    method: str = "pearson",
    threshold: float = 0.7,
    plot: bool = True,
    figsize: tuple = (12, 10)
) -> pd.DataFrame:
    """
    Compute and display correlation matrix for numeric columns.
    
    Args:
        df: DataFrame to analyze
        up_to_column: If provided, only include columns up to (and including) this column
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
        threshold: Highlight correlations above this threshold (in absolute value)
        plot: Whether to create a heatmap
        figsize: Figure size for the heatmap
    
    Returns:
        Correlation matrix as DataFrame
    """
    # Select columns up to specified column
    if up_to_column:
        if up_to_column not in df.columns:
            raise KeyError(f"Column '{up_to_column}' not found in dataframe")
        col_idx = df.columns.get_loc(up_to_column)
        df_subset = df.iloc[:, :col_idx + 1]
    else:
        df_subset = df
    
    # Select only numeric columns
    numeric_df = df_subset.select_dtypes(include=["number"])
    
    if numeric_df.empty:
        log_print("No numeric columns found for correlation analysis")
        return pd.DataFrame()
    
    log_print(f"\nComputing {method.capitalize()} correlation matrix for {len(numeric_df.columns)} numeric columns")
    log_print(f"Columns: {list(numeric_df.columns)[:10]}{'...' if len(numeric_df.columns) > 10 else ''}\n")
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Print summary statistics
    log_print("="*60)
    log_print("CORRELATION MATRIX SUMMARY")
    log_print("="*60)
    log_print(f"Matrix shape: {corr_matrix.shape}")
    log_print(f"Method: {method}")
    
    # Find high correlations (excluding diagonal)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False)
        log_print(f"\nHigh correlations (|r| >= {threshold}):")
        log_print(high_corr_df.to_string(index=False))
    else:
        log_print(f"\nNo correlations found with |r| >= {threshold}")
    
    # Print full correlation matrix
    # print("\n" + "="*60)
    # print("FULL CORRELATION MATRIX")
    # print("="*60)
    # print(corr_matrix.to_string())
    
    # Plot heatmap if requested
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle (optional - remove if you want full matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True if len(corr_matrix) <= 15 else False,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
            ax=ax
        )
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix',
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save and/or show based on global flags
        if SAVE_PLOTS:
            output_path = OUTPUT_DIR / "plots" / f"correlation_matrix_{method}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to: {output_path}")
        
        if SHOW_PLOTS:
            plt.show()
        
        if not SHOW_PLOTS:
            plt.close()
    
    return corr_matrix

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
        """Create a ColumnTransformer that scales numeric and one-hot encodes categorical columns."""
        # identify column types
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # For robustness, remove potential target from these lists at call site.
        numeric_pipeline = Pipeline(
                steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                ]
        )
        categorical_pipeline = Pipeline(
                steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
        )

        transformers = []
        if numeric_cols:
                transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols:
                transformers.append(("cat", categorical_pipeline, categorical_cols))

        if not transformers:
                # no transformers found; fallback to a passthrough
                return ColumnTransformer(transformers=[("pass", "passthrough", df.columns.tolist())])

        return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(model_name: str, **kwargs):
        """Return an instantiated estimator by name. Extendable."""
        name = model_name.lower()
        if name in ("rf", "random_forest"):
                return RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 200), random_state=kwargs.get("random_state", 0), n_jobs=kwargs.get("n_jobs", -1))
        if name in ("svm", "svr", "linearsvr"):
            from sklearn.svm import LinearSVR
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.preprocessing import StandardScaler
            
            base_svr = LinearSVR(
                C=1.0,
                epsilon=0.0,
                max_iter=10000,
                dual=False,
                random_state=kwargs.get("random_state", 0)
            )
            
            return TransformedTargetRegressor(
                regressor=base_svr,
                transformer=StandardScaler()
            )
        if name in ("linear", "linear_regression", "ols", "mlr"):
                from sklearn.linear_model import LinearRegression
                # Multiple Linear Regression / Ordinary Least Squares
                return LinearRegression(fit_intercept=kwargs.get("fit_intercept", True), n_jobs=kwargs.get("n_jobs", None))
        if name in ("gradient_boosting", "gbm", "gbregressor"):
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(n_estimators=kwargs.get("n_estimators", 200), learning_rate=kwargs.get("learning_rate", 0.1), random_state=kwargs.get("random_state", 0))
        if name in ("gpr", "gaussian_process", "gaussian"):
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
                
                # More flexible kernel with noise term
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0)
                
                return GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    alpha=1e-6,  # regularization
                    normalize_y=True  # normalize target internally
                )
        if name in ("multioutput_rf",):
                # wrap RF in a multi-output wrapper (works when target has multiple columns)
                base = RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 200), random_state=kwargs.get("random_state", 0), n_jobs=kwargs.get("n_jobs", -1))
                return MultiOutputRegressor(base, n_jobs=kwargs.get("n_jobs", None))
        if name in ("polynomial", "polynomial_regression", "poly"):
                # Polynomial regression pipeline: PolynomialFeatures -> Ridge (regularized linear)
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import make_pipeline
                degree = int(kwargs.get("degree", 2))
                alpha = float(kwargs.get("alpha", 1.0))
                return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha, random_state=kwargs.get("random_state", None)))
        if name in ("pce", "polynomial_chaos"):
                # fallback: use polynomial features + linear model as simple surrogate
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import make_pipeline
                degree = kwargs.get("degree", 2)
                return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=1.0))
        if name in ("extra_trees", "extra_trees_regressor"):
                from sklearn.ensemble import ExtraTreesRegressor
                return ExtraTreesRegressor(n_estimators=kwargs.get("n_estimators", 200), random_state=kwargs.get("random_state", 0), n_jobs=kwargs.get("n_jobs", -1))
        if name in ("mlp", "neural_network", "nn"):
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=kwargs.get("hidden_layers", (100, 50)),
                activation=kwargs.get("activation", "relu"),
                max_iter=kwargs.get("max_iter", 500),
                random_state=kwargs.get("random_state", 0),
                early_stopping=True
            )
        if name in ("xgboost", "xgb"):
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=kwargs.get("n_estimators", 200),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", 6),
                    random_state=kwargs.get("random_state", 0),
                    n_jobs=kwargs.get("n_jobs", -1)
                )
            except ImportError:
                raise ImportError("xgboost not installed: pip install xgboost")
        if name in ("lightgbm", "lgbm"):
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=kwargs.get("n_estimators", 200),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    random_state=kwargs.get("random_state", 0),
                    n_jobs=kwargs.get("n_jobs", -1)
                )
            except ImportError:
                raise ImportError("lightgbm not installed: pip install lightgbm")

        if name in ("catboost", "cat"):
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=kwargs.get("n_estimators", 200),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    random_state=kwargs.get("random_state", 0),
                    verbose=False
                )
            except ImportError:
                raise ImportError("catboost not installed: pip install catboost")
        raise ValueError(f"Unknown model_name: {model_name!r}")


def build_pipeline(preprocessor: ColumnTransformer, model_name: str = "random_forest", model_kwargs: dict | None = None) -> Pipeline:
    """Return a sklearn Pipeline combining preprocessor and chosen regressor/model."""
    model_kwargs = model_kwargs or {}
    model = make_model(model_name, **model_kwargs)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def train_test_pipeline(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 0,
        model_name: str = "random_forest",
) -> Tuple[Pipeline, Dict[str, float]]:
        """Train a surrogate model pipeline from a dataframe.

        Returns:
            - trained Pipeline
            - metrics dict with 'rmse' and 'r2' on test set
        """
        target = infer_target(df, target_col)
        X = df.drop(columns=[target])
        y = df[target].values

        if X.empty:
                raise ValueError("No feature columns available after dropping target")

        preprocessor = build_preprocessor(X)
        pipeline = build_pipeline(preprocessor, model_name=model_name)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))


        # diagnostics (run where you have y_test and y_pred)
        mean_y = float(np.mean(y_test))
        std_y = float(np.std(y_test))
        mae = float(mean_absolute_error(y_test, y_pred))
        # rmse already computed above
        nrmse = float(rmse / mean_y) if mean_y != 0 else float("nan")
        rmse_std = float(rmse / std_y) if std_y != 0 else float("nan")

        # define full metrics dict to return
        metrics = {
                "rmse": rmse,
                "r2": r2,
                "mean": mean_y,
                "std": std_y,
                "mae": mae,
                "nrmse": nrmse,
                "rmse_std": rmse_std,
        }

        # print brief diagnostics
        # print(f"mean(y_test) = {mean_y}")
        # print(f"std(y_test)  = {std_y}")
        # print(f"RMSE         = {rmse}")
        # print(f"NRMSE (RMSE/mean) = {nrmse}")
        # print(f"RMSE/std     = {rmse_std}")
        # print(f"MAE          = {mae}")


        return pipeline, metrics


def save_pipeline(pipeline: Pipeline, model_name: str, path: Union[str, Path] = OUTPUT_DIR, variable: str = TARGET_COLUMN) -> None:
        """Save the pipeline to disk using joblib."""
        path_obj = Path(path)
        
        # If path is a directory, create the filename; otherwise use parent directory
        if path_obj.is_dir():
                output_dir = path_obj
        else:
                output_dir = path_obj.parent
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with model name
        filename = f"surrogate_pipeline_{variable}_{model_name}.joblib"
        output_path = output_dir / filename
        joblib.dump(pipeline, output_path)

def train_from_csv(path: str = DEFAULT_CSV, target_col: Optional[str] = None) -> Tuple[Pipeline, Dict[str, float]]:
        """Convenience wrapper: load CSV, train pipeline, return (pipeline, metrics)."""
        df = load_csv(path)
        pipeline, metrics = train_test_pipeline(df, target_col=target_col)
        return pipeline, metrics


def evaluate_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    pipeline: Optional[Pipeline] = None,
    methods: list[str] = None,
    n_repeats: int = 10,
    random_state: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate feature importance/scoring using multiple methods.
    
    Args:
        df: DataFrame with features and target
        target_col: name of target column
        pipeline: Optional trained pipeline; if provided, extracts model-based importance
        methods: list of methods to use. Available:
            - "correlation_pearson": Pearson correlation with target
            - "correlation_spearman": Spearman correlation with target
            - "mutual_info": Mutual information (sklearn)
            - "model_importance": Model's native feature_importances_ (tree models)
            - "permutation": Permutation importance (works for any model)
        n_repeats: number of permutation repeats (for permutation importance)
        random_state: random seed
    
    Returns:
        Dictionary mapping method name -> DataFrame with columns ['feature', 'importance', 'std']
        (std is only present for permutation importance)
    """
    if methods is None:
        methods = ["correlation_pearson", "correlation_spearman", "mutual_info"]
        if pipeline is not None:
            methods.extend(["model_importance", "permutation"])
    
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Get feature names after preprocessing (if pipeline provided)
    feature_names = X.columns.tolist()
    if pipeline is not None:
        try:
            # Transform X to get preprocessed feature names
            X_transformed = pipeline.named_steps["preprocessor"].transform(X)
            # Try to get feature names from preprocessor
            if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
                feature_names_transformed = pipeline.named_steps["preprocessor"].get_feature_names_out()
            else:
                # fallback: use numeric indices
                feature_names_transformed = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        except Exception as e:
            warnings.warn(f"Could not extract transformed feature names: {e}")
            feature_names_transformed = feature_names
    
    results = {}
    
    # 1. Pearson correlation
    if "correlation_pearson" in methods:
        correlations = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                try:
                    corr, _ = pearsonr(X[col].fillna(X[col].mean()), y)
                    correlations.append({"feature": col, "importance": abs(corr)})
                except Exception:
                    correlations.append({"feature": col, "importance": 0.0})
            else:
                correlations.append({"feature": col, "importance": 0.0})
        results["correlation_pearson"] = pd.DataFrame(correlations).sort_values("importance", ascending=False)
    
    # 2. Spearman correlation
    if "correlation_spearman" in methods:
        correlations = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                try:
                    corr, _ = spearmanr(X[col].fillna(X[col].mean()), y, nan_policy="omit")
                    correlations.append({"feature": col, "importance": abs(corr)})
                except Exception:
                    correlations.append({"feature": col, "importance": 0.0})
            else:
                correlations.append({"feature": col, "importance": 0.0})
        results["correlation_spearman"] = pd.DataFrame(correlations).sort_values("importance", ascending=False)
    
    # 3. Mutual information
    if "mutual_info" in methods:
        from sklearn.feature_selection import mutual_info_regression
        # Only use numeric columns for mutual info (or encode categoricals first)
        X_numeric = X.select_dtypes(include=["number"]).fillna(X.select_dtypes(include=["number"]).mean())
        if not X_numeric.empty:
            mi_scores = mutual_info_regression(X_numeric, y, random_state=random_state)
            mi_df = pd.DataFrame({
                "feature": X_numeric.columns,
                "importance": mi_scores
            }).sort_values("importance", ascending=False)
            results["mutual_info"] = mi_df
    
    # 4. Model native feature importance (tree-based models)
    if "model_importance" in methods and pipeline is not None:
        regressor = pipeline.named_steps["regressor"]
        # Check if it's a nested pipeline (e.g., polynomial regression)
        if hasattr(regressor, "steps"):
            # It's a nested pipeline; get the last step
            regressor = regressor.steps[-1][1]
        
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            # Normalize to [0, 1] (sum to 1) for consistency
            importances = importances / importances.sum()
        
            imp_df = pd.DataFrame({
                "feature": [f.replace("num__", "").replace("cat__", "") for f in feature_names_transformed[:len(importances)]],
                "importance": importances
            }).sort_values("importance", ascending=False)
            results["model_importance"] = imp_df
        else:
            warnings.warn(f"Model {type(regressor).__name__} does not have feature_importances_ attribute")
    
    # 5. Permutation importance (works for any model)
    if "permutation" in methods and pipeline is not None:
        from sklearn.model_selection import train_test_split
        # Use a test split to avoid overfitting in importance estimates
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        # Speed up: use smaller sample and fewer repeats for large datasets
        max_samples_perm = 5000
        if len(X_test) > max_samples_perm:
            X_test = X_test.sample(n=max_samples_perm, random_state=random_state)
            y_test = y_test[X_test.index] if hasattr(y_test, 'index') else y_test[:max_samples_perm]
        
        perm_importance = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1  # limit parallelism to avoid resource exhaustion
        )
        
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance": perm_importance.importances_mean,
            "std": perm_importance.importances_std
        }).sort_values("importance", ascending=False)
        results["permutation"] = perm_df
    
    # 6. SHAP importance (if method included and pipeline provided)
    if "shap" in methods and pipeline is not None:
        try:
            import shap
            regressor = pipeline.named_steps["regressor"]
            if hasattr(regressor, "steps"):
                regressor = regressor.steps[-1][1]
            
            # Get transformed features
            X_transformed = pipeline.named_steps["preprocessor"].transform(X)
            
            # SHAP only works well with tree models
            if hasattr(regressor, "feature_importances_"):
                explainer = shap.TreeExplainer(regressor)
                shap_values = explainer.shap_values(X_transformed)
                
                # Mean absolute SHAP as importance
                shap_importance = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({
                    "feature": [f.replace("num__", "").replace("cat__", "") 
                               for f in feature_names_transformed[:len(shap_importance)]],
                    "importance": shap_importance
                }).sort_values("importance", ascending=False)
                results["shap"] = shap_df
        except ImportError:
            warnings.warn("shap not installed; skipping SHAP importance")
    
    return results


def print_feature_importance(importance_dict: Dict[str, pd.DataFrame], top_n: int = 20) -> None:
    """Pretty print feature importance results."""
    for method, df in importance_dict.items():
        log_print(f"\n{'='*60}")
        log_print(f"Feature Importance: {method.upper()}")
        log_print('='*60)
        display_df = df.head(top_n).copy()
        if "std" in display_df.columns:
            display_df["importance"] = display_df.apply(
                lambda row: f"{row['importance']:.4f} ± {row['std']:.4f}", axis=1
            )
            display_df = display_df[["feature", "importance"]]
        log_print(display_df.to_string(index=False))
        log_print()


def plot_feature_importance(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: tuple = (8, 8)
) -> None:
    """Plot feature importance as horizontal bar charts."""
    n_methods = len(importance_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method, df) in zip(axes, importance_dict.items()):
        top_features = df.head(top_n).copy()
        
        if "std" in top_features.columns:
            ax.barh(top_features["feature"], top_features["importance"], xerr=top_features["std"])
        else:
            ax.barh(top_features["feature"], top_features["importance"])
        
        ax.set_xlabel("Importance")
        ax.set_title(f"{method.replace('_', ' ').title()}")
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def compare_feature_importance(
    importance_original: Dict[str, pd.DataFrame],
    importance_model: Dict[str, pd.DataFrame],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare feature importance across different methods.
    
    Returns a merged DataFrame showing rankings/scores side-by-side.
    """
    # Start with a base method (e.g., Pearson correlation)
    base_method = "correlation_pearson"
    if base_method not in importance_original:
        base_method = list(importance_original.keys())[0]
    
    comparison_df = importance_original[base_method][["feature"]].copy()
    
    # Add all original data methods
    for method_name, df in importance_original.items():
        comparison_df = comparison_df.merge(
            df[["feature", "importance"]].rename(columns={"importance": f"{method_name}"}),
            on="feature",
            how="outer"
        )
    
    # Add all model methods
    for method_name, df in importance_model.items():
        # For model methods, feature names might be transformed (one-hot encoded)
        # Strip preprocessor prefixes like "num__" or "cat__" to match original feature names
        df_clean = df.copy()
        df_clean["feature"] = df_clean["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
        comparison_df = comparison_df.merge(
            df_clean[["feature", "importance"]].rename(columns={"importance": f"{method_name}"}),
            on="feature",
            how="left"
        )
    # Calculate average importance across methods (excluding NaNs)
    numeric_cols = [col for col in comparison_df.columns if col != "feature"]
    comparison_df["avg_importance"] = comparison_df[numeric_cols].mean(axis=1, skipna=True)
    
    # Sort by average importance
    comparison_df = comparison_df.sort_values("avg_importance", ascending=False)
    
    return comparison_df.head(top_n)


def format_model_name(model_name: str) -> str:
    """Convert model_name like 'extra_trees' to 'Extra Trees'."""
    return model_name.replace("_", " ").title()


def plot_comparison_feature_importance(
    importance_original: Dict[str, pd.DataFrame],
    importance_model: Dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: tuple = (10, 6),
    model_name: str = ""
) -> None:
    """Plot comparison of feature importance across original data and model methods."""
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })
    
    # Use compare_feature_importance to get the comparison DataFrame
    comp_df = compare_feature_importance(importance_original, importance_model, top_n=top_n)
    
    # Extract feature names and drop the avg_importance column for plotting
    features = comp_df["feature"].tolist()
    comp_df_plot = comp_df.drop(columns=["feature", "avg_importance"]).set_index(pd.Index(features))
    
    # Extract error bars from permutation importance if available
    permutation_errors = None
    if "permutation" in importance_model:
        perm_df = importance_model["permutation"]
        # Match features and extract std values
        perm_dict = dict(zip(perm_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True), 
                             perm_df["std"]))
        permutation_errors = [perm_dict.get(feat, 0) for feat in features]
    
    # Clean up column names
    method_labels = {
        "correlation_pearson": "Pearson Correlation (Data)",
        "correlation_spearman": "Spearman Correlation (Data)",
        "mutual_info": "Mutual Information (Data)",
        "model_importance": "Model Importance (Surrogate)",
        "permutation": "Permutation Importance (Surrogate)"
    }
    comp_df_plot.columns = [method_labels.get(col, col.replace("_", " ").title()) for col in comp_df_plot.columns]
    
    # Define fixed colors for each method type
    method_color_map = {
        "Pearson Correlation (Data)": "#4A90E2",
        "Spearman Correlation (Data)": "#357ABD",
        "Mutual Information (Data)": "#2E5F8F",
        "Model Importance (Surrogate)": "#FFA442",
        "Permutation Importance (Surrogate)": "#E66322"
    }
    
    # Map colors to the actual columns in comp_df_plot
    colors = [method_color_map.get(col, "#999999") for col in comp_df_plot.columns]
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare error bars - only for permutation importance column
    xerr = None
    if permutation_errors and "Permutation Importance (Surrogate)" in comp_df_plot.columns:
        perm_col_idx = list(comp_df_plot.columns).index("Permutation Importance (Surrogate)")
        xerr = [[0] * len(comp_df_plot.columns) for _ in range(len(features))]
        for i in range(len(features)):
            xerr[i][perm_col_idx] = permutation_errors[i]
        xerr = np.array(xerr).T
    
    comp_df_plot.plot(kind="barh", ax=ax, color=colors, width=0.75, edgecolor='black', 
                      linewidth=0.5, xerr=xerr, error_kw={'elinewidth': 0.5, 'capsize': 2, 'alpha': 0.6})
    
    ax.set_xlabel("Importance Score", fontweight='normal')
    ax.set_ylabel("Feature", fontweight='normal')
    
    # Main title
    fig.suptitle(f"Feature Importance: Statistical Methods vs. {format_model_name(model_name)} Model", 
                 fontweight='bold', fontsize=12, y=0.98)
    
    # Add subtitle with target variable name
    fig.text(0.5, 0.94, TARGET_COLUMN.replace('_', ' ').replace('.', ' ').title(), 
             ha='center', va='top', fontsize=10, fontstyle='italic', color='#555555',
             transform=fig.transFigure)
    
    # Improve legend
    ax.legend(loc="lower right", fontsize=9, frameon=True, fancybox=False, 
              edgecolor='black', framealpha=0.9)
    
    # Add subtle grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save and/or show based on global flags
    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / "plots" / f"feature_importance_comparison_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance comparison to: {output_path}")
    
    if SHOW_PLOTS:
        plt.show()
    
    if not SHOW_PLOTS:
        plt.close()


def plot_parity_and_residuals(
    df: pd.DataFrame,
    pipeline: Pipeline,
    target_col: str,
    model_name: str,
    metrics: dict,
    figsize: tuple = (16, 6)
) -> None:
    """
    Plot parity plot and residuals histogram for model evaluation.
    
    Args:
        df: DataFrame with features and target
        pipeline: Trained pipeline
        target_col: Name of target column
        model_name: Name of the model (for title and filename)
        metrics: Dictionary containing pre-calculated metrics (rmse, r2, mae, nrmse, etc.)
        figsize: Figure size tuple (width, height)
    """
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })
    
    target = infer_target(df, target_col)
    X = df.drop(columns=[target])
    y = df[target].values
    y_pred = pipeline.predict(X)
    
    # Extract metrics from the passed dictionary
    r2 = metrics['r2']
    rmse = metrics['rmse']
    mae = metrics['mae']
    nrmse = metrics['nrmse'] * 100  # convert to percentage
    
    # Create figure with adjusted width ratios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                     gridspec_kw={'width_ratios': [1.2, 1]})  # Give left plot more space
    
    # === LEFT PANEL: Parity Plot ===
    # Scatter plot with transparency
    ax1.scatter(y, y_pred, s=20, alpha=0.5, edgecolors='none', color='#2E86AB', label='Predictions')
    
    # Perfect prediction line (y=x) - use axis limits for true diagonal
    lims = [
        min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
        max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ]
    ax1.plot(lims, lims, 'k--', linewidth=1.5, 
             label='Perfect Prediction', zorder=10)
    
    # Add ±20% error bounds (optional - adjust or remove as needed)
    margin = 0.20  # 20% error
    ax1.fill_between(lims, 
                     [lims[0] * (1 - margin), lims[1] * (1 - margin)],
                     [lims[0] * (1 + margin), lims[1] * (1 + margin)],
                     color='gray', alpha=0.15, label='±20% Error')
    
    # Set equal limits and aspect ratio for diagonal line
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal', adjustable='box')
    
    # Labels and formatting
    ax1.set_xlabel('Observed Values', fontweight='semibold')
    ax1.set_ylabel('Predicted Values', fontweight='semibold')
    ax1.set_title('(a) Parity Plot', fontweight='bold', loc='left')
    
    # Add metrics text box using pre-calculated values
    textstr = '\n'.join([
        f'$R^2$ = {r2:.4f}',
        f'RMSE = {rmse:.2e}',
        f'MAE = {mae:.2e}',
        f'NRMSE = {nrmse:.2f}%'
    ])
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=1)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Grid and legend
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', frameon=True, fancybox=False, 
               edgecolor='black', framealpha=0.9)
    
    # Clean up spines (add this for consistency with right panel)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # === RIGHT PANEL: Residuals Distribution ===
    residuals = y - y_pred

    # Create symmetric bins centered on zero
    max_abs_residual = max(abs(residuals.min()), abs(residuals.max()))
    bins = np.linspace(-max_abs_residual, max_abs_residual, 50)

    # Histogram with better styling
    n, bins, patches = ax2.hist(residuals, bins=bins, density=True, 
                                 color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Overlay normal distribution for reference
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_norm, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_norm - mu)/sigma)**2),
             'r--', linewidth=2, label=f'Normal($\mu$={mu:.2e}, $\sigma$={sigma:.2e})')
    
    # Vertical line at zero
    ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Zero Error')
    
    # Labels and formatting
    ax2.set_xlabel('Residuals (Observed - Predicted)', fontweight='semibold')
    ax2.set_ylabel('Probability Density', fontweight='semibold')
    ax2.set_title('(b) Residuals Distribution', fontweight='bold', loc='left')
    
    # Add statistics text box
    textstr_res = '\n'.join([
        f'Mean = {mu:.2e}',
        f'Std Dev = {sigma:.2e}',
        f'Skewness = {pd.Series(residuals).skew():.3f}',
        f'Kurtosis = {pd.Series(residuals).kurtosis():.3f}'
    ])
    props_res = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=1)
    ax2.text(0.95, 0.95, textstr_res, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props_res, family='monospace')
    
    # Grid and legend
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.legend(loc='upper left', frameon=True, fancybox=False, 
               edgecolor='black', framealpha=0.9)
    
    # Clean up spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Overall figure title
    fig.suptitle(f'{format_model_name(model_name)} Surrogate Model Performance', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add subtitle with target variable name
    fig.text(0.5, 0.94, TARGET_COLUMN.replace('_', ' ').replace('.', ' ').title(), 
             ha='center', va='top', fontsize=11, fontstyle='italic', color='#555555',
             transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save and/or show based on global flags
    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / "plots" / f"parity_residuals_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved parity/residuals plot to: {output_path}")
    
    if SHOW_PLOTS:
        plt.show()
    
    if not SHOW_PLOTS:
        plt.close()


def train_surrogate_model(model_name: str = MODEL_NAME, show_plot: bool = True, skip_correlation: bool = False, print_feature_importance_individually: bool = False) -> None:
    # Quick CLI behavior: train and save model, print metrics.
    parser = argparse.ArgumentParser(description="Train a surrogate model from CSV")
    parser.add_argument("--csv", "-c", default=DEFAULT_CSV, help="Path to results CSV")
    parser.add_argument("--target", "-t", default=None, help="Name of target column (default: last column)")
    parser.add_argument("--out", "-o", default=DEFAULT_MODEL_OUT, help="Output path for saved pipeline")

    def _parse_omit(value: str) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if value == "":
            return None
        else:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')
            for p in parts:
                if not pattern.match(p):
                    raise argparse.ArgumentTypeError(f"invalid column name in --omit: {p!r}")
            return ",".join(parts)

    parser.add_argument(
        "--omit",
        "-x",
        type=_parse_omit,
        default=None,
        help="Comma-separated list of column names to omit from training (e.g. --omit col1,col2)",
    )
    args = parser.parse_args()
    if args.target is None:
        if isinstance(TARGET_COLUMN, int):
            args.target = load_csv(args.csv).columns[TARGET_COLUMN]
        elif isinstance(TARGET_COLUMN, str):
            args.target = TARGET_COLUMN
        else:
                raise ValueError("TARGET_COLUMN must be an int or str")

    # Load csv
    df = load_csv(args.csv)
        
    # Drop all irrelevant output columns, other than the target column
    df = omit_columns(df, args.target)

    # Drop correlated features BEFORE training
    correlated_features_to_drop = ["commuting_jobs_share"]
    if any(col in df.columns for col in correlated_features_to_drop):
        if not skip_correlation:
            log_print(f"\nDropping correlated features: {[c for c in correlated_features_to_drop if c in df.columns]}")
        df = df.drop(columns=[c for c in correlated_features_to_drop if c in df.columns])

    timing_start = time()
    print(f"Training {model_name} surrogate model...")
    pipeline, metrics = train_test_pipeline(df, target_col=args.target, model_name=model_name)
    timing_end = time()
    metrics["training_time"] = timing_end - timing_start
    save_pipeline(pipeline, model_name, args.out)
    
    importance_original = evaluate_feature_importance(
        df, 
        target_col=args.target,
        methods=["correlation_pearson", "correlation_spearman", "mutual_info"]
    )
    importance_model = evaluate_feature_importance(
            df,
            target_col=args.target,
            pipeline=pipeline,
            methods=["model_importance", "permutation"]
        )
    
    if print_feature_importance_individually:
        # Feature importance evaluation
        log_print("\n" + "="*60)
        log_print("FEATURE IMPORTANCE ANALYSIS")
        log_print("="*60)
        
        # Print evaluation original data
        log_print("\n--- Original Data (Statistical Methods) ---")
        print_feature_importance(importance_original, top_n=15)
        
        # Print evaluation surrogate model
        log_print(f"\n--- Surrogate Model: {model_name} ---")
        print_feature_importance(importance_model, top_n=15)
    
    # Print side-by-side comparison
    log_print("\n" + "="*60)
    log_print(f"FEATURE IMPORTANCE COMPARISON: {model_name}")
    log_print("="*60)
    comparison_df = compare_feature_importance(importance_original, importance_model, top_n=20)
    log_print(comparison_df.to_string(index=False))
    log_print(f"Metrics on holdout: NRMSE={metrics['nrmse']*100:.2f}%, R2={metrics['r2']:.6g}")
    
    if show_plot:
        # Plot side-by-side comparison of feature importance in horizontal bar plot
        plot_comparison_feature_importance(importance_original, importance_model, top_n=15, model_name=model_name)
        
        # Parity plot and residuals histogram in one figure
        plot_parity_and_residuals(df, pipeline, target_col=args.target, model_name=model_name, metrics=metrics)

    # Show CSV columns and suggest how to pick a specific target next time.
    if args.target is None:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        log_print(f"Columns in CSV ({len(cols)}): {cols}")
        if numeric_cols:
            log_print(f"Numeric columns (likely targets): {numeric_cols}")

        if args.target is None:
            log_print("No --target provided: the script used the last column as the target by default.")
            log_print("To select one specific target column next time, re-run with: --target NAME")
    return metrics


def train_multiple_surrogate_models(model_names: list[str] = MODEL_NAMES) -> None:
    """Train multiple surrogate models and compare performance."""
        
    log_print("\n" + "="*60)
    log_print("TRAINING MULTIPLE MODELS")
    log_print("="*60)
    
    # Train all models and collect results
    results = {}
    for model_name in model_names:
        log_print(f"\n{'='*60}")
        log_print(f"{format_model_name(model_name)}")
        log_print(f"{'='*60}")
        
        results[model_name] = train_surrogate_model(model_name=model_name, show_plot=SHOW_PLOTS, skip_correlation=True, print_feature_importance_individually=False)
    
    # Print summary
    log_print(f"\n{'='*60}")
    log_print(f"SUMMARY: {TARGET_COLUMN.replace('_', ' ').title()}")
    log_print(f"{'='*60}")
    
    # Sort by NRMSE (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['nrmse'])
    
    for model_name, metrics in sorted_results:
        log_print(f"{format_model_name(model_name):20s}: NRMSE={metrics['nrmse']*100:5.2f}%, R2={metrics['r2']:.6f}, Time={metrics.get('training_time', 0):.2f}s")


def main(model_or_models):
    # Initialize logging
    init_log()
    
    # Run correlation analysis once before training all models
    df = load_csv(DEFAULT_CSV)
    
    log_print("\n" + "="*60)
    log_print("CORRELATION ANALYSIS")
    log_print("="*60)
    corr_matrix = analyze_correlations(
        df,
        up_to_column="blankenburgverbinding",
        method="pearson",
        threshold=0.7,
        plot=SHOW_PLOTS
    )

    log_print(f"\nTarget column: {TARGET_COLUMN}")
    log_print(f"Pipeline to be saved at: {DEFAULT_MODEL_OUT}")

    # Train either one or multiple surrogate models
    if isinstance(model_or_models, str):
        train_surrogate_model(model_name=model_or_models)
    elif len(model_or_models) == 1:
        train_surrogate_model(model_name=model_or_models[0])
    else:
        train_multiple_surrogate_models()
    close_log()

if __name__ == "__main__":
    main(models)

