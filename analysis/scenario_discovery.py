from pathlib import Path
import ema_workbench
import pandas as pd
import numpy as np
from typing import Any
import seaborn as sns

CURR_DIR = Path(__file__).parent
output_median = 5749882233.360854  # Median of combined_vkm.csv_year


def get_array_stats(array) -> dict[str, Any]:
    stats = {
        "mean": np.mean(array),
        "median": np.median(array),
        "std_dev": np.std(array),
        "min": np.min(array),
        "max": np.max(array),
        "lower_quartile": np.percentile(array, 25),
        "upper_quartile": np.percentile(array, 75),
    }
    return stats


def load_data():
    data = pd.read_csv(CURR_DIR / "ema_road_model_27_05_2024_results.csv", index_col=0)
    simulation_input_df = data.iloc[:, :-3]
    simulation_output_dict = data.iloc[:, -3:].to_dict()
    simulation_output_dict = {
        key: np.array(list(value.values()))
        for key, value in simulation_output_dict.items()
    }  # Convert the dictionary values to numpy arrays
    return simulation_input_df, simulation_output_dict


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
        numerical_array = np.where(numerical_array < threshold, 1, 0)

    return numerical_array


def prim(independent_var_df: pd.DataFrame, dependent_var_array: np.array):
    if not isinstance(independent_var_df, pd.DataFrame):
        raise TypeError(
            f"independent_var_df must be a pandas DataFrame, but is {type(independent_var_df)}"
        )
    if not isinstance(dependent_var_array, np.ndarray):
        raise TypeError(
            f"dependent_var_array must be a numpy array, but is {type(dependent_var_array)}"
        )

    dependent_var_array = convert_continuous_to_categorical(
        dependent_var_array, ..., "less"
    )  # find new value for threshold

    prim_obj = ema_workbench.analysis.prim.Prim(
        independent_var_df,
        dependent_var_array,
        threshold=0.2,
    )

    box = prim_obj.find_box()

    box.show_tradeoff()
    box.show_pairs_scatter()
    # box.inspect(5)

    # for _ in range(5):
    #     prim_obj.find_box()
    #     print(prim_obj.stats[0])

    pass


def cart(simulation_input: pd.DataFrame, simulation_output: np.array):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")

    result = ema_workbench.analysis.cart.CART(
        simulation_input, simulation_output, mass_min=0.05, mode="regression"
    )
    result = result.stats
    return result


def logistic_regression(simulation_input: pd.DataFrame, simulation_output: np.array):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError(
            f"simulation_input must be a pandas DataFrame, but is {type(simulation_input)}"
        )
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError(
            f"simulation_output must be a numpy array, but is {type(simulation_output)}"
        )
    if not np.all(np.isin(simulation_output, [0, 1])):
        simulation_output_stats = get_array_stats(simulation_output)
        lower_quartile = simulation_output_stats["lower_quartile"]

        simulation_output_binary = convert_continuous_to_categorical(
            simulation_output, lower_quartile, "less"
        )

    lr_object = ema_workbench.analysis.logistic_regression.Logit(
        simulation_input, simulation_output_binary
    )
    lr_object.run()
    lr_object.inspect(1)
    sns.set_style("white")
    lr_object.plot_pairwise_scatter(1)
    # lr_object.show_tradeoff()
    # lr_object.inspect(0)
    pass
    # lr_object.inspect(5)


def main():
    simulation_input, simulation_output = load_data()
    logistic_regression(simulation_input, simulation_output["combined_vkm.csv_year"])
    # cart(simulation_input, simulation_output["combined_vkm.csv_year"])
    # prim(simulation_input, simulation_output["combined_vkm.csv_year"])


if __name__ == "__main__":
    main()
