import copy
import pandas as pd
import os
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pathlib
from .tape_creator_functions import (
    create_lists_sampling_input,
    latin_hypercube_sampling,
    cubic_spline_interpolation_without_dict_transformation,
)
from .global_parameters_sampling import read_csv_file, clear_df

CURR_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = CURR_DIR.parents[1] / "data" / "init_data_EMA"


def populate_dataframe_interpolated(interpolated_values: dict) -> dict:
    # Retrieve df and empty to populate
    file_path = CURR_DIR.parents[1] / "data" / "init_data" / "missed_interpolated.csv"
    df_main = read_csv_file(file_path, index_col=0, header=0)
    df_main = clear_df(df_main)

    # Create a new dictionary to store DataFrames with unique names
    scenario_dfs = {}

    # Fill dataframe of each scenario
    for scenario, variables in interpolated_values.items():
        df_scenario = pd.DataFrame({"_year": df_main["_year"]})
        for variable, values in variables.items():
            df_scenario[variable] = values

        scenario_dfs[scenario] = df_scenario

    return scenario_dfs


def save_global_parameters_scenarios_as_csv(
    final_scenarios_dfs: dict,
    output_path: pathlib.Path = OUTPUT_DIR,
):
    """
    Save the DataFrames to CSV files
    """
    for scenario, df in final_scenarios_dfs.items():
        file_path = pathlib.Path(
            output_path / f"{scenario.lower()}_global_parameters"
        ).with_suffix(".csv")
        df.to_csv(file_path, index=True, header=True, sep=",", decimal=".")

def create_interpolation_input_dict(
    variables_list_global_params: list,
    base_values_2019: dict,
    sampled_values: np.ndarray,
) -> dict:
    number_of_scenarios = len(sampled_values)
    num_digits = len(str(number_of_scenarios))
    variables_to_interpolate = [
        var.replace("_2050", "")
        for var in variables_list_global_params
        if "_2050" in var
    ]

    interpolation_input_dict = {}
    if any("_2030" in var for var in variables_list_global_params):
        print("At least one variable contains '_2030'")
        indices_2030 = [
            i for i, var in enumerate(variables_list_global_params) if "_2030" in var
        ]
        indices_2050 = [
            i for i, var in enumerate(variables_list_global_params) if "_2050" in var
        ]

        for scenario_idx, scenario_values in enumerate(sampled_values):
            scenario_key = f"Scenario_{scenario_idx:0{num_digits}d}"
            interpolation_input_dict[scenario_key] = {}
            for i, variable_name in enumerate(variables_to_interpolate):
                value_2019 = base_values_2019.get(f"{variable_name}_2019", np.nan)
                interpolation_input_dict[scenario_key][variable_name] = [
                    value_2019,
                    scenario_values[indices_2030[i]],
                    scenario_values[indices_2050[i]],
                ]
    else:
        print("No variables contain '_2030'")
        # TODO: Write logic for when no '_2030' variables are present
    return interpolation_input_dict, variables_to_interpolate



def create_global_parameters_scenarios(
    variables_list_global_params: list,
    sampled_values,
    base_values_2019: dict,
    output_path: pathlib.Path = OUTPUT_DIR):
    """
    The main function to create X number of global parameter scenarions based on the number of samples provided.
    Take the
    Interpolate the values between 2019 and 2030 and between 2030 and 2050 using cubic spline interpolation.
    The dataframes are stored as csv files in the provided path folder.
    """
    # Interpolate the values between 2019 and 2030 and between 2030 and 2050
    # TODO: implement properly
    
    interpolation_input_dict, variables_to_interpolate = create_interpolation_input_dict(
        variables_list_global_params=variables_list_global_params,
        base_values_2019=base_values_2019,
        sampled_values=sampled_values,
    )

    interpolated_values = cubic_spline_interpolation_without_dict_transformation(
        samples_dict=interpolation_input_dict, columns=variables_to_interpolate
    )

    final_scenarios_dfs = populate_dataframe_interpolated(interpolated_values)
    save_global_parameters_scenarios_as_csv(final_scenarios_dfs, output_path)


def main():
    create_global_parameters_scenarios()


if __name__ == "__main__":
    from tape_creator_functions import (
        create_lists_sampling_input,
        latin_hypercube_sampling,
        cubic_spline_interpolation,
    )

    main()
# else:
#     from tape_creator_functions import (
#     create_lists_sampling_input,
#     latin_hypercube_sampling,
#     cubic_spline_interpolation,
# )
#     main()
