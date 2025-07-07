# Import the necessary modules
import pathlib
import pandas as pd
from scipy.stats import qmc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats.qmc import discrepancy

# from global_parameters_sampling import (
#     create_global_parameters_scenarios,
# )
# from local_parameters_tape_creator import create_local_parameters_scenarios

# User sets desired number of scenarios here
CURR_DIR = pathlib.Path(__file__).parent
NUMBER_OF_SCENARIOS = 10
OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data_EMA"
RANDOM_SEED_NUMBER = 0


def load_bandwith_data(file_path: str):
    """
    Load bandwidth data from an excel file.
    """
    bandwiths = {}
    base_values_2019 = {}

    # Define the path to the Excel file
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    # Load the Excel file
    try:
        # Load all sheets from the Excel file
        bandwiths_file = pd.read_excel(
            file_path, sheet_name=0, index_col=0, na_values=["n/a"]
        )
    except ImportError:
        raise ImportError("Pandas is required to load the bandwidth data.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")
    # Placeholder for actual bandwidth data loading logic

    ## Fill the dictionary with the bandwiths data
    # Start with elasticities
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("elasticity_min")) and pd.notnull(
            row.get("elasticity_max")
        ):
            var_name = f"{row.get('var')}_elasticity"
            if pd.notnull(var_name):
                bandwiths[var_name] = [row["elasticity_min"], row["elasticity_max"]]
    # Get bandwiths of input values 2030
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("2030_min")) and pd.notnull(row.get("2030_max")):
            var_name = f"{row.get('var')}_2030"
            if pd.notnull(var_name):
                bandwiths[var_name] = [row["2030_min"], row["2030_max"]]
    # Get bandwiths of input values 2050
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("2050_min")) and pd.notnull(row.get("2050_max")):
            var_name = f"{row.get('var')}_2050"
            if pd.notnull(var_name):
                bandwiths[var_name] = [row["2050_min"], row["2050_max"]]

    # Create a list of the base year values 2019
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("2050_min")) and pd.notnull(row.get("2050_max")):
            var_name = f"{row.get('var')}_2019"
            if pd.notnull(var_name):
                base_values_2019[var_name] = row[2019]

    print("Bandwidth data loaded.")

    # Create a new dictionary with the required structure
    bandwiths_summary = {
        "Variable_names": list(bandwiths.keys()),
        "lower_bounds": [v[0] for v in bandwiths.values()],
        "upper_bounds": [v[1] for v in bandwiths.values()],
    }

    return bandwiths_summary, base_values_2019


def evaluate_lhs_quality(samples: np.ndarray) -> dict:
    """
    Evaluate space-filling quality of a Latin Hypercube Sample.

    Parameters:
        samples (np.ndarray): A 2D array of shape (n_samples, n_dimensions)
                              from a qmc.LatinHypercube or similar sampler.

    Returns:
        dict: Dictionary with quality metrics.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    # Pairwise distances
    pairwise_dists = pdist(samples, metric="euclidean")
    pairwise_sq = squareform(pairwise_dists)

    # Minimum and mean distances
    min_dist = np.min(pairwise_dists)
    mean_dist = np.mean(pairwise_dists)

    # Fill distance: max distance from any point to its nearest neighbor
    fill_dist = np.max(np.min(pairwise_sq + np.eye(len(samples)) * 1e10, axis=1))

    # Correlation
    corr = np.corrcoef(samples, rowvar=False)
    mean_abs_corr = np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)]))

    # L2-star discrepancy (using scipy.stats.qmc.discrepancy)
    try:
        raw_l2_star_discrepancy = discrepancy(samples, method="L2-star")
    except Exception:
        raw_l2_star_discrepancy = np.nan

    def expected_l2_discrepancy(n, d, num_trials=100):
        total = 0.0
        for _ in range(num_trials):
            X = np.random.rand(n, d)
            total += discrepancy(X, method="L2-star")
        return total / num_trials

    expected_l2_star_discrepancy = expected_l2_discrepancy(
        n=len(samples), d=samples.shape[1]
    )
    normalized_l2_star_discrepancy = (
        raw_l2_star_discrepancy / expected_l2_star_discrepancy
    )

    # Audze-Eglais criterion: sum of inverse squared distances
    # (excluding self-distances)
    n = samples.shape[0]
    # Avoid division by zero on diagonal
    mask = ~np.eye(n, dtype=bool)
    audze_eglais = np.sum(1.0 / (pairwise_sq[mask] ** 2))

    return {
        "min_pairwise_distance": min_dist,
        "mean_pairwise_distance": mean_dist,
        "fill_distance": fill_dist,
        "mean_abs_correlation": mean_abs_corr,
        "l2_star_discrepancy": raw_l2_star_discrepancy,
        "normalized_l2_star_discrepancy": normalized_l2_star_discrepancy,
        "audze_eglais": audze_eglais,
    }




def sample_elasticities_and_params(
    bandwiths_dict: dict,
    number_of_scenarios: int = NUMBER_OF_SCENARIOS,
    include_2030: bool = True,
    random_seed_number: int = RANDOM_SEED_NUMBER,
):
    """
    Sample elasticities and parameters for the EMA simulation.
    This function is a placeholder for the actual implementation.
    """
    if not include_2030:
        # Remove all variables ending with "_2030" from the dictionary fields
        mask = [not name.endswith("_2030") for name in bandwiths_dict["Variable_names"]]
        for key in ["Variable_names", "lower_bounds", "upper_bounds"]:
            bandwiths_dict[key] = [
                v for v, keep in zip(bandwiths_dict[key], mask) if keep
            ]
    # Extract the variable names and bounds
    variable_names = bandwiths_dict["Variable_names"]
    l_bounds = bandwiths_dict["lower_bounds"]
    u_bounds = bandwiths_dict["upper_bounds"]
    assert (
        len(variable_names) == len(l_bounds) == len(u_bounds)
    ), "Variable names, lower bounds, and upper bounds must have the same length."

    sampler = qmc.LatinHypercube(
        d=len(l_bounds),
        seed=random_seed_number,
        # optimization="lloyd"
    )
    samples = sampler.random(n=number_of_scenarios)

    # Evaluate the quality of the Latin Hypercube Sample
    quality_metrics = evaluate_lhs_quality(samples)
    print(
        f"Quality metrics of the Latin Hypercube Sample (2030 dropped = {not include_2030}):"
    )
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Scale the samples to the bounds
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
    return variable_names, scaled_samples


def main(number_of_scenarios: int = NUMBER_OF_SCENARIOS, include_2030: bool = True):
    sampling_input_data = load_bandwith_data(
        CURR_DIR.parent.joinpath("EMA_input_sample_paper_2.xlsx")
    )
    variable_names, sampled_values = sample_elasticities_and_params(
        sampling_input_data, number_of_scenarios, include_2030=include_2030
    )
    return variable_names, sampled_values


if __name__ == "__main__":
    main()
