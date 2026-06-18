# Import the necessary modules
import pathlib
import numpy as np
import pandas as pd
from .global_parameters_sampling_paper_3 import (
    create_global_parameters_scenarios,
)
from .local_parameters_tape_creator_paper_3 import create_local_parameters_scenarios
from .global_parameters_sampling import clear_df
import shutil


# Init
OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data_EMA"
RAILWAY_SHEET_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data"
SECONDS_LIST = [
        0,
        31536000,
        63072000,
        94608000,
        126144000,
        157680000,
        189216000,
        220752000,
        252288000,
        283824000,
        315360000,
        346896000,
        378432000,
        409968000,
        441504000,
        473040000,
        504576000,
        536112000,
        567648000,
        599184000,
        630720000,
        662256000,
        693792000,
        725328000,
        756864000,
        788400000,
        819936000,
        851472000,
        883008000,
        914544000,
        946080000,
        977616000,
    ]

def create_parameter_files_multimodal(
    variables_list_global_params: list,
    sampled_values: np.ndarray,
    base_values_2019,
    output_path: pathlib.Path = OUTPUT_PATH,
    lloyd_optimization: bool = False,
):
    """ This function creates the parameter files for the simulations.
     It takes the variable names, the sampled values, the base values for 2019, and the output path as input.
     It creates the parameter files for the global parameters (csv), local parameters (json), railway kpi files (csv), traffic kpi (files), waterway (kpi).
     The global parameters are created using the create_global_parameters_scenarios function, which takes the variable names, the sampled values, the base values for 2019, and the output path as input.
     The local parameters are created using the create_local_parameters_scenarios function, which takes the variable names, the sampled values, and the output path as input.
     The kpi files are created using the create_kpi_files function, which takes the variable names, the sampled values, and the output path as input.
     The function also checks if the output folder is empty before creating the parameter files. If it is not empty, it asks the user if they want to clear it before proceeding.
    """

    # Check if the output folder is empty
    output_folder_empty_check(output_path)

    # 1. Create global scenarios from samples
    create_global_parameters_scenarios(
        variables_list_global_params, sampled_values, base_values_2019, output_path
    )
    print("Global parameters created")

    # 2. Create kpi files from samples
    create_kpi_files(
        variables_list_global_params, sampled_values, base_values_2019, output_path
    )
    print("KPI files created")

    # 3. Create local scenarios from samples
    create_local_parameters_scenarios(
        variables_list_global_params, sampled_values, output_path
    )
    print("Local parameters created")
    print("All parameter files created successfully")


def output_folder_empty_check(output_path: pathlib.Path) -> bool:
    # Check if the folder in OUTPUT_PATH is empty
    number_of_files_in_output_path_folder = len(list(output_path.glob("*")))
    if number_of_files_in_output_path_folder == 0:
        print(
            "Output folder checked and no files found. Proceeding to create parameter files."
        )
        return
    else:
        print(
            f"Output folder ({output_path}) is not empty, but contains {number_of_files_in_output_path_folder} files. Do you want to clear it?"
        )
        user_input = input("Type 'yes' to clear the folder or 'no' to exit: ")
        if user_input.lower() == "yes":
            for file in output_path.glob("*"):
                if file.is_file():
                    file.unlink()
            print("Output folder cleared")
            return

def create_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path):
    """
    This function creates the KPI files necessary for the (multimodal) simulation to compute emission and energy consumption. 
    The KPI files are created based on the sampled values and the base values for 2019.
    Three types of files are created: railway KPI files, traffic KPI files, and waterway KPI files.
    The KPI files are created in the output folder specified in the OUTPUT_PATH variable.
    """
    # Create railway KPI files
    create_railway_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path)

    # Create traffic KPI files
    create_traffic_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path)

    # Create waterway KPI files
    create_waterway_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path)


def interpolate_matching_columns_from_samples(
    df: pd.DataFrame,
    variables_list_global_params,
    sampled_values,
    base_values_2019,
    columns_offset: int = 2,
    sampled_row_idx: int = 0,
):
    """
    Interpolate matching KPI columns from base values (start) to sampled values (end).
    Returns the updated DataFrame and a mapping of matched column indices.
    """
    matched_column_indices = {
        column_name: variables_list_global_params.index(f"{column_name}_2050")
        for column_name in df.columns[columns_offset:]
        if f"{column_name}_2050" in variables_list_global_params
    }

    # Track found and not found columns
    found_columns = list(matched_column_indices.keys())
    not_found_columns = [
        column_name for column_name in df.columns[columns_offset:]
        if f"{column_name}_2050" not in variables_list_global_params
    ]
    
    print(f"Found {len(found_columns)} matching columns: {found_columns}")
    print(f"Not found {len(not_found_columns)} matching columns: {not_found_columns}")

    n_rows = len(df)
    sampled_values_arr = np.asarray(sampled_values)

    # Clear only columns that have a matching sampled variable.
    # Columns without a match remain untouched.
    for column_name in matched_column_indices:
        df[column_name] = np.nan


    for column_name, idx in matched_column_indices.items():
        if isinstance(base_values_2019, dict):
            start_value = base_values_2019.get(column_name, None)
        else:
            try:
                start_value = base_values_2019[idx]
            except (IndexError, TypeError, KeyError):
                continue

        try:
            if sampled_values_arr.ndim == 1:
                end_value = sampled_values_arr[idx]
            elif sampled_values_arr.ndim >= 2:
                end_value = sampled_values_arr[sampled_row_idx, idx]
            else:
                continue
        except (IndexError, TypeError):
            continue

        if pd.isna(start_value) or pd.isna(end_value):
            continue

        df[column_name] = np.linspace(float(start_value), float(end_value), n_rows)

    return df


def create_railway_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path):
    """
    This function creates the railway KPI files necessary for the (multimodal) simulation to compute emission and energy consumption. 
    The KPI files are created based on the sampled values and the base values for 2019.
    The KPI files are created in the output folder specified in the OUTPUT_PATH variable.
    """
    # Create railway KPI files
    energy_sources_rail = ["diesel", "electricity", "h2"]
    railway_kpi_rapid_development = {}

    for energy_source in energy_sources_rail:
        railway_kpi_rapid_development[energy_source] = pd.read_csv(
            RAILWAY_SHEET_PATH / f"railway_kpi_coefficients_{energy_source}_rapid_development.csv"
        )
        print(railway_kpi_rapid_development[energy_source].columns)

    # Compare first two dataframes in the dictionary
    if len(railway_kpi_rapid_development) >= 2:
        first_key, second_key = list(railway_kpi_rapid_development.keys())[:2]
        first_df = railway_kpi_rapid_development[first_key]
        second_df = railway_kpi_rapid_development[second_key]

        first_columns = list(first_df.columns)
        second_columns = list(second_df.columns)

        missing_in_second = [col for col in first_columns if col not in second_columns]
        missing_in_first = [col for col in second_columns if col not in first_columns]

        if not missing_in_second and not missing_in_first:
            print(f"Column names match exactly for '{first_key}' and '{second_key}'.")
        else:
            print(f"Column name mismatch between '{first_key}' and '{second_key}'.")
            print(f"Columns only in '{first_key}': {missing_in_second}")
            print(f"Columns only in '{second_key}': {missing_in_first}")

        common_columns = [col for col in first_columns if col in second_columns]
        exact_match_columns = []
        different_value_columns = []

        for col in common_columns:
            if first_df[col].equals(second_df[col]):
                exact_match_columns.append(col)
            else:
                different_value_columns.append(col)

        print(f"Compared '{first_key}' vs '{second_key}'")
        print(f"Exact match columns ({len(exact_match_columns)}): {exact_match_columns}")
        print(f"Different value columns ({len(different_value_columns)}): {different_value_columns}")

    

    for energy_source, railway_kpi_df in railway_kpi_rapid_development.items():
        railway_kpi_rapid_development[energy_source] = interpolate_matching_columns_from_samples(
            railway_kpi_df,
            variables_list_global_params,
            sampled_values,
            base_values_2019,
            columns_offset=2, # Seconds and years
            sampled_row_idx=0,
        )

    # railway_kpi_rapid_development.to_csv(os.path.join(output_path, "railway_kpi_rapid_development.csv"), index=False)


def create_traffic_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path):
    """
    This function creates the traffic KPI files necessary for the (multimodal) simulation to compute emission and energy consumption. 
    The KPI files are created based on the sampled values and the base values for 2019.
    The KPI files are created in the output folder specified in the OUTPUT_PATH variable.
    """
    # Create traffic KPI files
    pass  # Replace with actual implementation

def create_waterway_kpi_files(variables_list_global_params, sampled_values, base_values_2019, output_path):
    """
    This function creates the waterway KPI files necessary for the (multimodal) simulation to compute emission and energy consumption. 
    The KPI files are created based on the sampled values and the base values for 2019.
    The KPI files are created in the output folder specified in the OUTPUT_PATH variable.
    """
    # Create waterway KPI files
    pass  # Replace with actual implementation



def main():
    create_parameter_files_multimodal()
    create_scenario_config_files()


if __name__ == "__main__":
    main()
