# Import the necessary modules
import pathlib
from global_parameters_sampling import create_global_parameters_scenarios
from local_parameters_tape_creator import create_local_parameters_scenarios

# User sets desired number of scenarios here
number_of_scenarios = 12
output_path = pathlib.Path(__file__).parents[1] / "data" / "init_data_EMA"

def create_scenarios():
    # Sample global scenarios
    create_global_parameters_scenarios(number_of_scenarios, output_path)
    # Sample local scenarios
    create_local_parameters_scenarios(number_of_scenarios, output_path)

if __name__ == "__main__":
    create_scenarios()