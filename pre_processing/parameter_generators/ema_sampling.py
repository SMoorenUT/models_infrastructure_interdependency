# Import the necessary modules
import pathlib
import pandas as pd
from scipy.stats import qmc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats.qmc import discrepancy
import ast
import warnings

# from global_parameters_sampling import (
#     create_global_parameters_scenarios,
# )
# from local_parameters_tape_creator import create_local_parameters_scenarios

# User sets desired number of scenarios here
CURR_DIR = pathlib.Path(__file__).parent
NUMBER_OF_SCENARIOS = 10
OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data_EMA"
RANDOM_SEED_NUMBER = 0


def _parse_list_like_value(value):
    """Parse list-like strings such as '[-1.1, -0.1]' into Python lists."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def create_elasticity_mask(bandwiths_file, variable_names):
    """
    Create a mask for the elasticities to know which elasticities are part of which model.
    [passenger, cargo_domestic, cargo_international]

    Returns:
        a dictionary with the elasticities mask
    """
    variable_names = [var for var in variable_names if var.endswith("_elasticity")]
    models = ["passenger", "cargo_domestic", "cargo_international"]
    elasticity_mask = {}
    # Initialize the mask for each model
    for model in models:
        mask = []
        for var in variable_names:
            # Find the row in bandwiths_file where 'var' matches the current variable
            var_name = var.removesuffix("_elasticity")
            matching_rows = bandwiths_file[
                bandwiths_file["var"].astype(str).str.contains(var_name)
            ]
            # TODO Hard code GDP as special case
            if not matching_rows.empty and matching_rows.iloc[0][model] == 1:
                if var.startswith("gdp") and (model in var):
                    mask.append(True)
                elif var.startswith("gdp") and not (model in var):
                    mask.append(False)
                else:
                    mask.append(True)
            else:
                mask.append(False)
        elasticity_mask[model] = mask
    return elasticity_mask


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
            if "[" in var_name and "]" in var_name:
                var_name = var_name.replace("_elasticity", "")
                part1 = (
                    var_name[var_name.find("[") + 1 : var_name.find(",")].strip()
                    + "_elasticity"
                )
                part2 = (
                    var_name[var_name.find(",") + 1 : var_name.find("]")].strip()
                    + "_elasticity"
                )
                row["elasticity_min"] = ast.literal_eval(row["elasticity_min"])
                row["elasticity_max"] = ast.literal_eval(row["elasticity_max"])
                bandwiths[part1] = [row["elasticity_min"][0], row["elasticity_max"][0]]
                bandwiths[part2] = [row["elasticity_min"][1], row["elasticity_max"][1]]
            elif pd.notnull(var_name):
                bandwiths[var_name] = [row["elasticity_min"], row["elasticity_max"]]
    # Get bandwiths of input values 2030
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("2030_min")) and pd.notnull(row.get("2030_max")):
            var_name = f"{row.get('var')}_2030"
            if var_name.startswith("[gdp"):
                # Special case for GDP, which is a list of two values
                bandwiths[f"gdp_2030"] = [row["2030_min"], row["2030_max"]]
            elif pd.notnull(var_name):
                bandwiths[var_name] = [row["2030_min"], row["2030_max"]]
    # Get bandwiths of input values 2050
    for idx, row in bandwiths_file.iterrows():
        if pd.notnull(row.get("2050_min")) and pd.notnull(row.get("2050_max")):
            var_name = f"{row.get('var')}_2050"
            if var_name.startswith("[gdp"):
                # Special case for GDP, which is a list of two values
                bandwiths[f"gdp_2050"] = [row["2050_min"], row["2050_max"]]
            elif pd.notnull(var_name):
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

    elasticity_mask = create_elasticity_mask(
        bandwiths_file, bandwiths_summary["Variable_names"]
    )

    return bandwiths_summary, base_values_2019, elasticity_mask

def load_bandwith_data_multimodal(file_path: str):
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
            file_path, sheet_name=0, index_col=0, na_values=[-999, "-999"]
        )
    except ImportError:
        raise ImportError("Pandas is required to load the bandwidth data.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")

    modal_cols = [col for col in bandwiths_file.columns if any(m in col for m in ("road", "rail", "waterway"))]

    ## Fill the dictionary with the bandwiths data
    # Start with elasticities
    for idx, row in bandwiths_file.iterrows():
        if any(pd.notnull(row.get(col)) for col in modal_cols):
            for col in modal_cols:
                var_name = f"{row.get('var')}_elasticity_{col}"
                bandwiths[var_name] = _parse_list_like_value(row[col])

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
                base_values_2019[var_name] = row["2019"]

    print("Bandwidth data loaded.")

    # Keep only entries with valid (non-NaN) bounds
    valid_bandwiths = { # a list of bandwiths with values that are not NaN and are lists of length 2
        k: v
        for k, v in bandwiths.items()
        if isinstance(v, list)
        and len(v) >= 2
        and pd.notnull(v[0])
        and pd.notnull(v[1])
    }

    # Create a new dictionary with the required structure
    bandwiths_summary = {
        "Variable_names": list(valid_bandwiths.keys()),
        "lower_bounds": [v[0] for v in valid_bandwiths.values()],
        "upper_bounds": [v[1] for v in valid_bandwiths.values()],
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
    lloyd_optimization: bool = False,
    print_evaluation: bool = False,
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
    # Create a boolean mask for variables where lower bound >= upper bound
    invalid_bounds_mask = [l >= u for l, u in zip(l_bounds, u_bounds)]
    variables_not_to_sample = []
    for name, l, u in zip(variable_names, l_bounds, u_bounds):
        if l >= u:
            variables_not_to_sample.append(name)
    warnings.warn(
        f"({len(variables_not_to_sample)} variables have not been sampled. {variables_not_to_sample}"
    )

    sampler = qmc.LatinHypercube(
        d=len(l_bounds) - sum(invalid_bounds_mask),
        seed=random_seed_number,
        optimization="lloyd" if lloyd_optimization else None,
    )
    samples = sampler.random(
        n=number_of_scenarios // 2
    )  # Half the number of scenarios because it will be doubled over policy

    # Evaluate the quality of the Latin Hypercube Sample
    quality_metrics = evaluate_lhs_quality(samples)
    if print_evaluation:
        # Print the quality metrics
        print(
            f"Quality metrics of the Latin Hypercube Sample (2030 dropped = {not include_2030}):"
        )
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Scale the samples to the bounds
    # Only use valid bounds for sampling/scaling
    valid_l_bounds = [l for l, valid in zip(l_bounds, invalid_bounds_mask) if not valid]
    valid_u_bounds = [u for u, valid in zip(u_bounds, invalid_bounds_mask) if not valid]
    variable_names_sampled = [
        name for name, valid in zip(variable_names, invalid_bounds_mask) if not valid
    ]
    scaled_samples = qmc.scale(samples, valid_l_bounds, valid_u_bounds)

    variable_names_to_return = variable_names_sampled.copy()
    scenario_input_values = scaled_samples.copy()
    for idx, var_name in enumerate(variable_names):
        if var_name not in variable_names_sampled:
            variable_names_to_return.append(var_name)
            new_col = np.full(
                (number_of_scenarios // 2, 1), bandwiths_dict["lower_bounds"][idx]
            )
            scenario_input_values = np.hstack([scenario_input_values, new_col])

    assert (
        len(variable_names_to_return)
        == scenario_input_values.shape[1]
        == len(variable_names)
    ), "Variable names and scaled samples must have the same number of columns."
    return variable_names_to_return, scenario_input_values


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
