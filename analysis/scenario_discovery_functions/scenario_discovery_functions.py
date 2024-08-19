
import numpy as np
import pandas as pd
import ema_workbench

def calculate_basic_statistics(data: np.ndarray or list) -> dict: # type: ignore
    if not isinstance(data, (np.ndarray, list)):
        raise TypeError("data must be a numpy array or list")

    statistics = {
        "mean": np.mean(data),
        "median": np.median(data),
        "lower_quartile": np.percentile(data, 25),
        "upper_quartile": np.percentile(data, 75),
        "lower_quintile": np.percentile(data, 20),
        "upper_quintile": np.percentile(data, 80),
        "lower_decile": np.percentile(data, 10),
        "upper_decile": np.percentile(data, 90),
        "minimum": np.min(data),
        "maximum": np.max(data),
        "standard_deviation": np.std(data),
        "variance": np.var(data),
    }

    return statistics


def convert_continuous_to_categorical(
    numerical_array: np.array, threshold: float, criterion: str
):

    # Convert the numerical array to a binary array based on the threshold and criterion

    # Check the input types
    criterion = criterion.lower()
    criterion_values = ["greater", "less", "greater_or_equal", "less_or_equal", "equal"]
    # TODO: Implement the 'between' criterion
    if not isinstance(numerical_array, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")
    if not isinstance(threshold, float):
        raise TypeError("threshold must be a float")
    if not isinstance(criterion, str):
        raise TypeError("criterion must be a string")
    if criterion not in criterion_values:
        raise ValueError(
            f"criterion must be one of the following: {', '.join(criterion_values)}, but criterion is {criterion}"
        )

    # Convert the numerical array to a binary array based on input threshold and criterion
    if criterion == "greater":
        numerical_array = np.where(numerical_array > threshold, 1, 0)
    elif criterion == "less":
        numerical_array = np.where(numerical_array < threshold, 1, 0)
    elif criterion == "greater_or_equal":
        numerical_array = np.where(numerical_array >= threshold, 1, 0)
    elif criterion == "less_or_equal":
        numerical_array = np.where(numerical_array <= threshold, 1, 0)
    elif criterion == "equal":
        numerical_array = np.where(numerical_array == threshold, 1, 0)
    else:
        Warning("numerical_array is not converted to a binary array")

    return numerical_array


def prim(independent_var_df: pd.DataFrame, dependent_var_array: np.array) -> tuple:
    if not isinstance(independent_var_df, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(dependent_var_array, np.ndarray):
        raise TypeError("dependent_var_array must be a numpy array")

    lower_quartile = calculate_basic_statistics(dependent_var_array)["lower_quartile"]
    dependent_var_array = convert_continuous_to_categorical(
        dependent_var_array, lower_quartile, "less"
    )

    prim_obj = ema_workbench.analysis.prim.Prim(
        independent_var_df,
        dependent_var_array,
        threshold=0.8,
    )

    box = prim_obj.find_box()

    box.show_tradeoff()

    box.show_pairs_scatter()

    return prim_obj, box

    


def cart(simulation_input: pd.DataFrame, simulation_output: np.array):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")

    result = ema_workbench.analysis.cart.CART(
        simulation_input, simulation_output, mass_min=0.05, mode="regression"
    )
    return result.stats


def logistic_regression(
    simulation_input: pd.DataFrame, 
    simulation_output: np.array, 
    limit_section_of_interest_name: str = None, 
    direction_section_of_interest: str = None
):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")

    # Convert the continuous output to a binary output if needed
    if limit_section_of_interest_name and direction_section_of_interest and not np.all(np.isin(simulation_output, [0, 1])): # Check if the output is already binary and arguments are provided to make binary
        simulation_output_stats = calculate_basic_statistics(simulation_output)
        if limit_section_of_interest_name not in simulation_output_stats:
            raise ValueError("limit_section_of_interest_name must be one of the keys from the statistics")
        limit_section_of_interest_value = simulation_output_stats[limit_section_of_interest_name]
        simulation_output = convert_continuous_to_categorical(simulation_output, limit_section_of_interest_value, direction_section_of_interest)


    lr_object = ema_workbench.analysis.logistic_regression.Logit(
        simulation_input, simulation_output, threshold=0.5
    )
    lr_object.run()
    return lr_object