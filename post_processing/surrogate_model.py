from __future__ import annotations

from sklearn import metrics
import surrogate_model
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

"""
surrogate_model.py

Utilities for training, evaluating and saving simple surrogate regression pipelines
from CSV results. Intended for quick experimentation and lightweight production use.

Key features:
- Load CSV and infer or accept a specified target column.
- Omit designated output columns (configured via NUMBER_OF_OUTPUT_COLUMNS_IN_CSV).
- Robust preprocessing: numeric imputation + scaling, categorical imputation + one-hot.
- Flexible model factory: rf, extra_trees, svr, gaussian_process, multioutput_rf, polynomial_chaos, etc.
- Builds a sklearn Pipeline (preprocessor + regressor), trains with train/test split.
- Evaluation: RMSE, R2, MAE, NRMSE, RMSE/std, plus parity and residual plots.
- Save/load pipelines with joblib for later inference.
- Small CLI wrapper (train_surrogate_model) for one-shot training and saving.

Defaults and common functions:
- DEFAULT_CSV, DEFAULT_MODEL_OUT, NUMBER_OF_OUTPUT_COLUMNS_IN_CSV, TARGET_COLUMN, MODEL_NAME
- programmatic API: load_csv, omit_columns, infer_target, build_preprocessor, make_model,
        build_pipeline, train_test_pipeline, train_from_csv, save_pipeline, load_pipeline, predict_from_df

CLI usage examples:
                python surrogate_model.py
                python surrogate_model.py --csv data.csv --target Y --out model.joblib
                python surrogate_model.py --csv data.csv --omit col1,col2

Programmatic usage examples:
                df = load_csv("data.csv")
                pipeline, metrics = train_test_pipeline(df, target_col="Y", model_name="extra_trees")
                save_pipeline(pipeline, "surrogate.joblib")
"""


DEFAULT_CSV = "/home/moorens/code/analysis/ema_road_model_17_07_2025_results.csv"
DEFAULT_MODEL_OUT = "/home/moorens/code/analysis/surrogate_pipeline.joblib"

NUMBER_OF_OUTPUT_COLUMNS_IN_CSV = 6  # Example: last 6 columns are outputs
TARGET_COLUMN = -4  # Example: last column is the main target

MODEL_NAME = "gradient_boosting"
SHOW_PLOTS = False

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


def make_model(model_name: str = "random_forest", **kwargs):
        """Return an instantiated estimator by name. Extendable."""
        name = model_name.lower()
        if name in ("rf", "random_forest"):
                return RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 200), random_state=kwargs.get("random_state", 0), n_jobs=kwargs.get("n_jobs", -1))
        if name in ("svm", "svr"):
                return SVR(C=kwargs.get("C", 1.0), kernel=kwargs.get("kernel", "rbf"))
        if name in ("linear", "linear_regression", "ols", "mlr"):
                from sklearn.linear_model import LinearRegression
                # Multiple Linear Regression / Ordinary Least Squares
                return LinearRegression(fit_intercept=kwargs.get("fit_intercept", True), n_jobs=kwargs.get("n_jobs", None))
        if name in ("gradient_boosting", "gbm", "gbregressor"):
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(n_estimators=kwargs.get("n_estimators", 200), learning_rate=kwargs.get("learning_rate", 0.1), random_state=kwargs.get("random_state", 0))
        if name in ("gpr", "gaussian_process"):
                from sklearn.gaussian_process import GaussianProcessRegressor
                return GaussianProcessRegressor()  # tune kernel / alpha as needed (single-output)
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


def save_pipeline(pipeline: Pipeline, path: str = DEFAULT_MODEL_OUT) -> None:
        """Save the pipeline to disk using joblib."""
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        joblib.dump(pipeline, path)


def load_pipeline(path: str = DEFAULT_MODEL_OUT) -> Pipeline:
        """Load a pipeline previously saved with save_pipeline."""
        if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)


def predict_from_df(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
        """Return predictions for the supplied dataframe (features only)."""
        return pipeline.predict(df)


def train_from_csv(path: str = DEFAULT_CSV, target_col: Optional[str] = None) -> Tuple[Pipeline, Dict[str, float]]:
        """Convenience wrapper: load CSV, train pipeline, return (pipeline, metrics)."""
        df = load_csv(path)
        pipeline, metrics = train_test_pipeline(df, target_col=target_col)
        return pipeline, metrics


def train_surrogate_model(model_name: str = MODEL_NAME, show_plot: bool = True) -> None:
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


    print(f"Loading CSV: {args.csv}")
    df = load_csv(args.csv)
    df = omit_columns(df, args.target)
    print(f"Using target column: {infer_target(df, args.target)}")
    print(f"Training {model_name} surrogate model...")
    pipeline, metrics = train_test_pipeline(df, target_col=args.target, model_name=model_name)
    print(f"Metrics on holdout: NRMSE={metrics['nrmse']*100:.2f}%, R2={metrics['r2']:.6g}")
    save_pipeline(pipeline, args.out)
    print(f"Saved pipeline to: {args.out}")

    if show_plot:
        # Additional evaluation: Parity plot and residuals histogram
        target = infer_target(df, args.target)
        X = df.drop(columns=[target])
        y = df[target].values
        y_pred = pipeline.predict(X)

        plt.scatter(y, y_pred, s=6, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.title(f'Parity - {model_name}')
        plt.figure()
        plt.hist(y - y_pred, bins=50)
        plt.title(f'Residuals - {model_name}')
        plt.show()

    # Show CSV columns and suggest how to pick a specific target next time.
    if args.target is None:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        print(f"Columns in CSV ({len(cols)}): {cols}")
        if numeric_cols:
            print(f"Numeric columns (likely targets): {numeric_cols}")

        if args.target is None:
            print("No --target provided: the script used the last column as the target by default.")
            print("To select one specific target column next time, re-run with: --target NAME")
    return metrics


def train_multiple_surrogate_models():
        model_names = ["random_forest", "extra_trees", "svr", "gradient_boosting", "linear"]
        results = {}
        for model_name in model_names:
                results[model_name] = train_surrogate_model(model_name=model_name, show_plot=SHOW_PLOTS)
        print("\nSummary of all models:")
        for model_name, metrics in results.items():
                print(f"{model_name}: NRMSE={metrics['nrmse']*100:.2f}%, R2={metrics['r2']:.6g}") 

if __name__ == "__main__":
      train_multiple_surrogate_models()
        # train_surrogate_model(model_name=MODEL_NAME)