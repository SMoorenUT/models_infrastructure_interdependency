import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from tqdm import tqdm

from train_surrogate_model import DEFAULT_MODEL_OUT

CUR_DIR = Path(__file__).parent
CSV_PATH = CUR_DIR / "surrogate_model_inputs_paper_2.csv"
PIPELINE_PATH = CUR_DIR / "surrogate_pipeline_gradient_boosting.joblib"
VARIABLE = "combined_vkt_year_2050"


def load_data(path: Path) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def load_pipeline(path: Path = PIPELINE_PATH) -> Pipeline:
    """Load a pipeline previously saved with save_pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_from_df(
    pipeline: Pipeline, df: pd.DataFrame, batch_size: int = 500
) -> np.ndarray:
    """
    Return predictions for the supplied dataframe with progress bar.

    Args:
        pipeline: Trained sklearn pipeline
        df: Input features DataFrame
        batch_size: Number of samples per batch (default 10000)

    Returns:
        Array of predictions
    """
    n_samples = len(df)

    # Small dataset - predict directly with simple progress indicator
    if n_samples <= 1000:
        print(f"Predicting {n_samples} samples...")
        predictions = pipeline.predict(df)
        print("Done!")
        return predictions

    # Large dataset - batch prediction with progress bar
    print(f"Predicting {n_samples:,} samples in batches of {batch_size:,}...")
    predictions = []

    for start_idx in tqdm(
        range(0, n_samples, batch_size), desc="Predicting", unit="batch", ncols=80
    ):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = df.iloc[start_idx:end_idx]
        batch_pred = pipeline.predict(batch)
        predictions.append(batch_pred)

    return np.concatenate(predictions)


def predict_from_csv(pipeline: Pipeline, path: Path) -> np.ndarray:
    """Load data from CSV and return predictions."""
    df = load_data(path)
    return predict_from_df(pipeline, df)


# Make a function to plot the distribution of predictions
def plot_prediction_distribution(predictions: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    # Set publication-quality style with smaller fonts
    plt.rcParams.update(
        {
            "font.size": 9,  # Reduced from 11
            "font.family": "serif",
            "axes.labelsize": 10,  # Reduced from 12
            "axes.titlesize": 11,  # Reduced from 12
            "xtick.labelsize": 8,  # Reduced from 10
            "ytick.labelsize": 8,  # Reduced from 10
            "legend.fontsize": 8,  # Reduced from 10
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot histogram with thinner edges
    n, bins, patches = ax.hist(
        predictions,
        bins=50,
        color="#2E86AB",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )  # Reduced from 0.5

    ax.set_xlabel(VARIABLE.replace("_", " "))
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Distribution of Surrogate Model Predictions", pad=8
    )  # Reduced padding

    # Add grid behind bars
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4, zorder=0)  # Thinner grid
    ax.set_axisbelow(True)

    # Add statistics annotation with smaller line width
    mean_val = np.mean(predictions)
    std_val = np.std(predictions)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.2,  # Reduced from 1.5
        label=f"Mean = {mean_val:.2e}",
    )  # Changed to scientific notation

    # Smaller legend
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", loc="upper right", fontsize=8
    )

    # Tighter layout with smaller margins
    plt.tight_layout(pad=0.5)
    plt.show()


def main():
    pipeline = load_pipeline(PIPELINE_PATH)
    df = load_data(CSV_PATH)
    predictions = predict_from_df(pipeline, df)

    # Append predictions as a new column to the dataframe and save to CSV
    df[VARIABLE] = predictions
    output_csv_path = (
        CUR_DIR.parents[1]
        / "analysis"
        / f"ema_road_model_17_07_2025_results_surrogate_model.csv"
    )
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved at {output_csv_path}")
    plot_prediction_distribution(predictions)


if __name__ == "__main__":
    main()
