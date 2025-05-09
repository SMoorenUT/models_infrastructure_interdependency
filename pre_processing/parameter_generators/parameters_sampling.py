# Import the necessary modules
import pathlib
from .global_parameters_sampling import (
    create_global_parameters_scenarios,
)
from .local_parameters_tape_creator import create_local_parameters_scenarios

# User sets desired number of scenarios here
NUMBER_OF_SCENARIOS = 10
OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "init_data_EMA"
RANDOM_SEED = 0  # Not implemented yet


def create_parameter_files(
    number_of_scenarios: int = NUMBER_OF_SCENARIOS,
    output_path: pathlib.Path = OUTPUT_PATH,
    random_seed: int = RANDOM_SEED,
):
    # Check if the output folder is empty
    output_folder_empty_check(output_path)
    # Sample global scenarios
    create_global_parameters_scenarios(number_of_scenarios, output_path, random_seed)
    print("Global parameters created")
    # Sample local scenarios
    create_local_parameters_scenarios(number_of_scenarios, output_path)
    print("Local parameters created")
    print("Parameter files created")


def create_scenario_config_files():
    pass  # to be implemented


def output_folder_empty_check(output_path) -> bool:
    # Check if the folder in OUTPUT_PATH is empty
    number_of_files_in_output_path_folder = len(list(output_path.glob("*")))
    if number_of_files_in_output_path_folder == 0:
        print("Output folder checked")
        return
    else:
        print(
            f"Output folder is not empty, but contains {number_of_files_in_output_path_folder} files. Do you want to clear it?"
        )
        user_input = input("Type 'yes' to clear the folder or 'no' to exit: ")
        if user_input == "yes":
            for file in output_path.glob("*"):
                file.unlink()
            print("Output folder cleared")
            return


def main():
    create_parameter_files()
    create_scenario_config_files()


if __name__ == "__main__":
    main()
