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
from scenario_generators import scenario_generator_ema

# Add the path to the current file
CURR_DIR = pathlib.Path(__file__).parent
NUMBER_OF_SCENARIOS = 10
OUTPUT_DIR = CURR_DIR.parent / "data"
INPUT_SHEET_DIR = CURR_DIR.joinpath("EMA_input_sample_paper_2.xlsx")
INCLUDE_2030 = True  # Whether to include 2030 input values in the sampling
RANDOM_SEED_NUMBER = 0
COMMUTING_JOBS_SHARE_2019 = 0.85  # Base value for commuting jobs share in 2019


# def add_commuting_jobs_share(variable_names, sampled_values, base_values_2019):
#     """ "
#     (Working_days - Remote_working_days * Percentage_of_remote_workers) / Working_days = Commuting_jobs_share
#     This function calculates the commuting_jobs_share based on the sampled values
#     and adds it to the variable_names and sampled_values lists.
#     """
#     # First for 2030 and 2050
#     for year in [2030, 2050]:
#         working_days = f"average_number_working_days_{year}"
#         remote_working_days = f"average_number_remote_working_days_{year}"
#         percentage_of_remote_workers = f"percentage_of_remote_workers_{year}"
#         commuting_jobs_share = f"commuting_jobs_share_{year}"

#         # Find index of the variables in the variable_names list
#         working_days_index = variable_names.index(working_days)
#         remote_working_days_index = variable_names.index(remote_working_days)
#         percentage_of_remote_workers_index = variable_names.index(
#             percentage_of_remote_workers
#         )

#         # Calculate the commuting jobs share
#         commuting_jobs_share_value = (
#             sampled_values[:, working_days_index]
#             - sampled_values[:, remote_working_days_index]
#             * (sampled_values[:, percentage_of_remote_workers_index] / 100)
#         ) / sampled_values[:, working_days_index]

#         # Add the commuting jobs share to the variable names and sampled values
#         variable_names.append(commuting_jobs_share)
#         sampled_values = np.column_stack((sampled_values, commuting_jobs_share_value))
#     # Add 2019 commuting_jobs_share
#     base_values_2019["commuting_jobs_share_2019"] = COMMUTING_JOBS_SHARE_2019

#     return variable_names, sampled_values, base_values_2019


def main():
    """Main function to create the parameter files and scenarios.
    This function will:

    1. Sample all values for elasticities and future input.
    2. Generate the scenarios based on the sampled parameters.
    """
    # Load bandwith data from the Excel file
    # The function returns a dictionary with the bandwiths and a dictionary with the base values
    # The bandwiths are the bounds for the elasticities and input values
    # The base values are the values for 2019, which are used as a reference
    # for the sampling
    sampling_input_data, base_values_2019 = ema_sampling.load_bandwith_data(
        CURR_DIR.joinpath("EMA_input_sample_paper_2.xlsx")
    )
    # Sample elasticities and parameters
    # The function returns the variable names and the sampled values
    variable_names, sampled_values = ema_sampling.sample_elasticities_and_params(
        sampling_input_data, NUMBER_OF_SCENARIOS, include_2030=INCLUDE_2030
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
        output_path=OUTPUT_DIR,
    )
    scenario_generator_ema.generate_and_output_multiple_scenarios(
        num_scenarios=NUMBER_OF_SCENARIOS, path=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
