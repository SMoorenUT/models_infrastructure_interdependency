"""
Set up everything necessary required for RDM simulation.
(1) Sample both input values and elasticities from here
(2) Create the multitude scenarios_config files, and global and local parameter files
"""

import numpy as np
import pandas as pd
import datetime as dt
import pathlib
from parameter_generators import parameters_sampling_paper_2 as parameters_sampling
from parameter_generators import ema_sampling
from parameter_generators.tape_creator_functions import add_commuting_jobs_share
from scenario_generators import scenario_generator_ema_paper_2 as scenario_generator_ema

# Constants and configurations
CURR_DIR = pathlib.Path(__file__).parent
NUMBER_OF_SCENARIOS = 1000
OUTPUT_DIR = CURR_DIR.parent / "data"
INPUT_SHEET_DIR = CURR_DIR.joinpath("EMA_input_sample_paper_2.xlsx")
INCLUDE_2030 = True  # Whether to include 2030 input values in the sampling
RANDOM_SEED_NUMBER = 0
COMMUTING_JOBS_SHARE_2019 = (
    0.92  # Share of commuting jobs in 2019, used as a base value
)
LLOYD_OPTIMIZATION = False  # Whether to use Lloyd optimization in the scenarios
LIST_OF_LOCAL_PARAMETERS = [
    "transport.shortest_path_lane_length",
    "jobs.count.index",
    "people.count.index",
    "transport.average_time.star",
]
PRINT_LHS_EVALUATION = (
    False  # Whether to print the evaluation of the Latin Hypercube Sampling
)


def main():
    """Main function to prepare simulation inputs and scenarios.

    This function will:
    1. Load input data and bandwidths from Excel.
    2. Sample elasticities and input parameters.
    3. Add commuting jobs share to the sampled data.
    4. Create parameter files for simulations.
    5. Generate and output multiple scenario configurations.
    """
    # Load bandwith data from the Excel file
    # The function returns a dictionary with the bandwiths and a dictionary with the base values
    # The bandwiths are the bounds for the elasticities and input values
    # The base values are the values for 2019, which are used as a reference
    # for the sampling
    sampling_input_data, base_values_2019, elasticity_mask = (
        ema_sampling.load_bandwith_data(
            CURR_DIR.joinpath("EMA_input_sample_paper_2.xlsx")
        )
    )
    # Sample elasticities and parameters
    # The function returns the variable names and the sampled values
    variable_names, sampled_values = ema_sampling.sample_elasticities_and_params(
        sampling_input_data,
        NUMBER_OF_SCENARIOS,
        include_2030=INCLUDE_2030,
        lloyd_optimization=LLOYD_OPTIMIZATION,
        print_evaluation=PRINT_LHS_EVALUATION,
    )
    # Add commuting_jobs_share
    variable_names, sampled_values, base_values_2019 = add_commuting_jobs_share(
        variable_names, sampled_values, base_values_2019, COMMUTING_JOBS_SHARE_2019
    )

    # Create parameter files
    # The function creates the parameter files in the output_path directory
    parameters_sampling.create_parameter_files(
        variables_list_global_params=variable_names,
        sampled_values=sampled_values,
        base_values_2019=base_values_2019,
        output_path=OUTPUT_DIR / "init_data_EMA",
    )
    scenario_generator_ema.generate_and_output_multiple_scenarios(
        variable_names=variable_names,
        sampled_values=sampled_values,
        elasticity_mask=elasticity_mask,
        output_path=OUTPUT_DIR / "scenarios_ema",
        local_parameters_to_drop=LIST_OF_LOCAL_PARAMETERS,
        lloyd_optimization_used=LLOYD_OPTIMIZATION,
        random_seed_number_used=RANDOM_SEED_NUMBER,
    )
    print("Simulation preparation completed successfully.")


if __name__ == "__main__":
    main()
