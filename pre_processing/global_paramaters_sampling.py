import pandas as pd
import os
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pathlib
from tape_creator_functions import (
    create_lists_sampling_input,
    latin_hypercube_sampling,
    cubic_spline_interpolation,
)

CURR_DIR = pathlib.Path(__file__).parent

variables_list_global_params = [
    "average_remote_working_days",
    "average_working_days",
    "percent_remote_worker",
    "fuel_price",
    "gdp",
    "share_service_sector_gdp",
    "total_vehicles",
    "world_trade_volume",
]


def clear_df(df):
    # Clear values except for the first two columns, headers, and the first row
    df.iloc[1:, 1:] = np.nan
    return df


def read_csv_file(file_path, index_col=None, header=None, delimiter=",", decimal="."):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(
        file_path,
        index_col=index_col,
        header=header,
        delimiter=delimiter,
        decimal=decimal,
    )
    return df


# Retrieve df and empty to populate
file_path = CURR_DIR.parent / "data" / "init_data" / "missed_interpolated.csv"
df_main = read_csv_file(file_path, index_col=0, header=0)
df_main = clear_df(df_main)

# Retrieve df to sample from
file_path = CURR_DIR / "input_data" / "EMA_sample_input.csv"
df_for_sampling_input = read_csv_file(
    file_path, index_col=0, header=0, delimiter=";", decimal=","
)


def create_sampling_dictionary(df_for_sampling_input):
    """
    Create a dictionary with keys representing years and values representing lists of sampling inputs.
    """
    sampling_dictionary = {
        "2019": create_lists_sampling_input(df_for_sampling_input, 2019),
        "2030_min": create_lists_sampling_input(df_for_sampling_input, 2030, "min"),
        "2030_max": create_lists_sampling_input(df_for_sampling_input, 2030, "max"),
        "2050_min": create_lists_sampling_input(df_for_sampling_input, 2050, "min"),
        "2050_max": create_lists_sampling_input(df_for_sampling_input, 2050, "max"),
    }
    return sampling_dictionary


def add_global_variable(
    variable_name: str, variable_values: list, sampling_dictionary: dict
):
    """ "
    Add a global variable to the list of global variables and its values to the dictionary of interpolated values.
    Variable values must have structure [2019, 2030_min, 2030_max, 2050_min, 2050_max].
    """
    # Check if the variable values has length 5
    if len(variable_values) != 5:
        raise ValueError(
            f"The variable values must have length 5, but has length {len(variable_values)}"
        )
    # Add the varibale name to variables_list_global_params
    variables_list_global_params.append(variable_name)
    # Iterate over the years and add the variable values to the sampling dictionary
    for i, year in enumerate(["2019", "2030_min", "2030_max", "2050_min", "2050_max"]):
        sampling_dictionary[year].append(variable_values[i])

    return sampling_dictionary


def populate_dataframe_interpolated(interpolated_values):
    # Create a new dictionary to store DataFrames with unique names
    scenario_dfs = {}

    # Fill dataframe of each scenario
    for scenario, variables in interpolated_values.items():
        df_scenario = pd.DataFrame({"_year": df_main["_year"]})
        variables_list = list(variables.keys())  # A list with all the variables
        for variable, values in variables.items():
            df_scenario[variable] = values

        scenario_dfs[scenario] = df_scenario

    return scenario_dfs


def save_global_parameters_scenarios_as_csv(final_scenarios_dfs):
    # Save the DataFrames to CSV files
    for scenario, df in final_scenarios_dfs.items():
        file_path = (
            CURR_DIR.parent / "data" / "init_data_EMA" / f"{scenario.lower()}.csv"
        )
        df.to_csv(file_path, index=True, header=True, sep=",", decimal=".")


def main():
    num_samples = 10
    sampling_dictionary = create_sampling_dictionary(df_for_sampling_input)
    sampling_dictionary = add_global_variable(
        "consumer_energy_price", [0.2, 0.1, 0.3, 0.1, 0.5], sampling_dictionary
    )
    sampling_dictionary = add_global_variable(
        "road_load_factor_index", [0.5, 0.4, 0.6, 0.35, 0.85], sampling_dictionary
    )
    sampling_dictionary = add_global_variable(
        "train_ticket_price", [100, 99, 102, 95, 106], sampling_dictionary
    )
    sampling_dictionary = add_global_variable(
        "commuting_jobs_share", [0.87, 0.75, 0.9, 0.5, 0.95], sampling_dictionary
    )


    lhs_samples_dict = latin_hypercube_sampling(sampling_dictionary, num_samples)

    interpolated_values = cubic_spline_interpolation(
        samples_dict=lhs_samples_dict, columns=variables_list_global_params
    )

    final_scenarios_dfs = populate_dataframe_interpolated(interpolated_values)
    save_global_parameters_scenarios_as_csv(final_scenarios_dfs)


if __name__ == "__main__":
    main()
