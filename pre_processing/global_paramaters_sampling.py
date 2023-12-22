import pandas as pd
import os
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pathlib

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
file_path = CURR_DIR / "data" / "init_data" / "missed_interpolated.csv"
df_main = read_csv_file(file_path, index_col=0, header=0)
df_main = clear_df(df_main)

# Retrieve df to sample from
file_path = CURR_DIR / "input_data" / "EMA_sample_input.csv"
df_for_sampling_input = read_csv_file(
    file_path, index_col=0, header=0, delimiter=";", decimal=","
)


# if df_main is not None:
# Print the DataFrame
# print("DataFrame:")
# print(df_main)


def create_lists_sampling_input(year, operator="min"):
    # Create lists from sampling df
    year_cols = {2019: [0], 2030: [1, 2], 2050: [3, 4]}

    if year not in year_cols:
        raise ValueError("Invalid year")

    cols = year_cols[year]

    if operator not in {"min", "max"}:
        raise ValueError("Invalid operator. Must be 'min' or 'max'")

    return getattr(df_for_sampling_input.iloc[:, cols], operator)(axis=1).tolist()


# Creating the dictionary
result_dict = {
    "2019": create_lists_sampling_input(2019),
    "2030_min": create_lists_sampling_input(2030, "min"),
    "2030_max": create_lists_sampling_input(2030, "max"),
    "2050_min": create_lists_sampling_input(2050, "min"),
    "2050_max": create_lists_sampling_input(2050, "max"),
}


def latin_hypercube_sampling():
    # Set the random seed for reproducibility
    np.random.seed(0)

    # Define lower and upper bounds
    lower_bound_2030 = result_dict["2030_min"]  # Lower bound for each dimension in 2030
    upper_bound_2030 = result_dict["2030_max"]  # Upper bound for each dimension in 2030
    lower_bound_2050 = result_dict["2050_min"]  # Lower bound for each dimension in 2050
    upper_bound_2050 = result_dict["2050_max"]  # Upper bound for each dimension in 2050

    # Define the number of samples and dimensions
    num_samples = 1000
    num_dimensions = (
        len(lower_bound_2030)
        if len(lower_bound_2030)
        == len(upper_bound_2030)
        == len(lower_bound_2050)
        == len(upper_bound_2050)
        else None
    )

    # Generate Latin Hypercube Samples in dictionary using scipy.stats.uniform
    lhs_samples_dict = {
        f"Scenario_{i:04d}": {
            "2019": result_dict["2019"],
            "2030": [
                uniform(
                    loc=lower_bound_2030[j],
                    scale=upper_bound_2030[j] - lower_bound_2030[j],
                ).rvs()
                for j in range(num_dimensions)
            ],
            "2050": [
                uniform(
                    loc=lower_bound_2050[j],
                    scale=upper_bound_2050[j] - lower_bound_2050[j],
                ).rvs()
                for j in range(num_dimensions)
            ],
        }
        for i in range(num_samples)
    }
    plt.show()

    return lhs_samples_dict


def cubic_spline_interpolation():
    # Create Data Points
    x = [2019, 2030, 2050]
    columns = [
        "average_remote_working_days",
        "average_working_days",
        "percent_remote_worker",
        "fuel_price",
        "gdp",
        "share_service_sector_gdp",
        "total_vehicles",
        "world_trade_volume",
    ]

    ## Transform the dictionary to {
    #   Scenario1:
    #       {Variable1: [2019value, 2030value, 2050value],
    #       Variable2: [2019value, 2030value, 2050value],
    #       VariableN:[..]
    #       },
    #   ScenarioN:
    #       {VariableN:...}
    #   }
    first_values_dict = {}

    for scen, years_dict in lhs_samples_dict.items():
        for year, values_list in years_dict.items():
            for position, key in enumerate(columns, start=1):
                if scen not in first_values_dict:
                    first_values_dict[scen] = {}

                if key not in first_values_dict[scen]:
                    first_values_dict[scen][key] = []

                first_values_dict[scen][key].append(values_list[position - 1])

    # Make list of scenario's to loop through (['Scenario_0001', 'Scenario_0002', ... , 'Scenario_XXXX'])
    scenarios = list(first_values_dict.keys())

    # List of column indices
    column_indices = np.arange(len(columns))

    # Dictionary to store cubic spline objects
    cs_dict = {}

    # Loop through scenarios
    for scenario in scenarios:
        # Create a nested dictionary for each scenario
        scenario_dict = {}

        # Loop through columns
        for column_index in column_indices:
            y = first_values_dict[scenario][columns[column_index]]

            # Perform Cubic Spline Interpolation
            cs = CubicSpline(x, y)

            # Evaluate Interpolation Function
            x_interp = list(range(2019, 2051))
            y_interp = cs(x_interp)

            column_name = columns[
                column_index
            ]  # Create a name for the variable (nested inside the scenario)
            scenario_dict[column_name] = y_interp

        # Assign the nested dictionary to the scenario key
        cs_dict[scenario] = scenario_dict

    return cs_dict


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


lhs_samples_dict = latin_hypercube_sampling()
# populate_dataframe()
interpolated_values = cubic_spline_interpolation()

final_scenarios_dfs = populate_dataframe_interpolated()
pass
