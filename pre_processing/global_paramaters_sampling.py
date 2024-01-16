import pandas as pd
import os
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pathlib
from tape_creator_functions import create_lists_sampling_input, latin_hypercube_sampling, cubic_spline_interpolation

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

CURR_DIR = pathlib.Path(__file__).parent

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

# Creating the dictionary
result_dict = {
    "2019": create_lists_sampling_input(df_for_sampling_input, 2019),
    "2030_min": create_lists_sampling_input(df_for_sampling_input, 2030, "min"),
    "2030_max": create_lists_sampling_input(df_for_sampling_input, 2030, "max"),
    "2050_min": create_lists_sampling_input(df_for_sampling_input, 2050, "min"),
    "2050_max": create_lists_sampling_input(df_for_sampling_input, 2050, "max"),
}
def populate_dataframe():
    # Create a new DataFrame for each scenario
    dfs = []

    # Fill dataframe of each scenario
    # for scenario, values in scenarios.items():
    #     df_scenario = pd.DataFrame({"_year": df_main["_year"]})
    #     columns = [
    #         "average_remote_working_days",
    #         "average_working_days",
    #         "percent_remote_worker",
    #         "fuel_price",
    #         "gdp",
    #         "share_service_sector_gdp",
    #         "total_vehicles",
    #         "world_trade_volume",
    #     ]
    #     df_scenario[columns] = np.nan
    #     for year, variable_values in values.items():
    #         if year in [2019, 2030, 2050]:
    #             for col, value in zip(columns, variable_values):
    #                 df_scenario.loc[df_scenario["_year"] == year, col] = value

    #     dfs.append(df_scenario)


def populate_dataframe_interpolated():
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


lhs_samples_dict = latin_hypercube_sampling(result_dict)
# populate_dataframe()
interpolated_values = cubic_spline_interpolation(samples_dict=lhs_samples_dict, columns=variables_list_global_params)

final_scenarios_dfs = populate_dataframe_interpolated()
pass
