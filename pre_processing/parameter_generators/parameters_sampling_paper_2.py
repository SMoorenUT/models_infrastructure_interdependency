# Import the necessary modules
import pathlib
import numpy as np
from .global_parameters_sampling_paper_2 import (
    create_global_parameters_scenarios,
)
from .local_parameters_tape_creator_paper_2 import create_local_parameters_scenarios
import shutil

# User sets desired number of scenarios here
OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data_EMA"


def create_parameter_files(
    variables_list_global_params: list,
    sampled_values: np.ndarray,
    base_values_2019,
    output_path: pathlib.Path = OUTPUT_PATH,
    lloyd_optimization: bool = False,
):
    # Check if the output folder is empty
    output_folder_empty_check(output_path)
    # Sample global scenarios
    create_global_parameters_scenarios(
        variables_list_global_params, sampled_values, base_values_2019, output_path
    )
    print("Global parameters created")
    # Sample local scenarios
    create_local_parameters_scenarios(
        variables_list_global_params, sampled_values, output_path
    )
    print("Local parameters created")
    print("Parameter files created")


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



def main():
    create_parameter_files()
    create_scenario_config_files()


if __name__ == "__main__":
    main()
