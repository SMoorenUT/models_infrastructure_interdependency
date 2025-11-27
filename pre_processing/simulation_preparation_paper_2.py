"""
Set up everything necessary required for RDM simulation.
(1) Sample both input values and elasticities from here
(2) Create the multitude scenarios_config files, and global and local parameter files
"""

import numpy as np
import pandas as pd
import datetime as dt
import pathlib
CURR_DIR = pathlib.Path(__file__).parent
from parameter_generators import parameters_sampling_paper_2 as parameters_sampling
from parameter_generators import ema_sampling
from parameter_generators.tape_creator_functions import add_commuting_jobs_share
import sys
sys.path.insert(0, str(CURR_DIR.parent))
from post_processing.analysis_preparation import generate_scenario_name_list
from scenario_generators import scenario_generator_ema_paper_2 as scenario_generator_ema

# Constants and configurations
NUMBER_OF_SCENARIOS = 10  # Total number of scenarios to generate
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
TARGET_OF_OUTPUT = "Original" # Target of the output, can be "Surrogate", "Original", or "Both"


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
    print("Starting LHS")
    variable_names, sampled_values = ema_sampling.sample_elasticities_and_params(
        sampling_input_data,
        NUMBER_OF_SCENARIOS,
        include_2030=INCLUDE_2030,
        lloyd_optimization=LLOYD_OPTIMIZATION,
        print_evaluation=PRINT_LHS_EVALUATION,
    )
    print("Finished LHS")
    # Add commuting_jobs_share
    years = [2030, 2050] if INCLUDE_2030 else [2050]
    variable_names, sampled_values, base_values_2019 = add_commuting_jobs_share(
        variable_names, sampled_values, base_values_2019, COMMUTING_JOBS_SHARE_2019, years=years
    )

    if TARGET_OF_OUTPUT in ["Original", "Both"]:
        output_dir = CURR_DIR.parent / "data"

        # Create parameter files
        # The function creates the parameter files in the output_path directory
        parameters_sampling.create_parameter_files(
            variables_list_global_params=variable_names,
            sampled_values=sampled_values,
            base_values_2019=base_values_2019,
            output_path=output_dir / "init_data_EMA",
        )
        scenario_generator_ema.generate_and_output_multiple_scenarios(
            variable_names=variable_names,
            sampled_values=sampled_values,
            elasticity_mask=elasticity_mask,
            output_path=output_dir / "scenarios_ema",
            local_parameters_to_drop=LIST_OF_LOCAL_PARAMETERS,
            lloyd_optimization_used=LLOYD_OPTIMIZATION,
            random_seed_number_used=RANDOM_SEED_NUMBER,
        )
    
    if TARGET_OF_OUTPUT in ["Surrogate", "Both"]:
        df = pd.DataFrame(sampled_values, columns=variable_names)
        
        # Remove unncessary columns
        df_2019 = df.filter(like="_2019")
        df_2030 = df.filter(like="_2030")
        df = df.drop(columns=df_2030.columns.tolist() + df_2019.columns.tolist())
        
        # Duplicate the scenarios over the policies
        df["blankenburgverbinding"] = 0
        df_duplicate = df.copy()
        df_duplicate["blankenburgverbinding"] = 1
        df = pd.concat([df, df_duplicate], ignore_index=True)

        # Remove from all column names the suffix _2050
        df.columns = [col.replace("_2050", "") for col in df.columns]

        # Remove _elasticty as a suffix from all column names and add elasticity_ as a prefix
        new_columns = []
        for col in df.columns:
            if col.endswith("_elasticity"):
                new_col = "elasticity_" + col[:-11]
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        df.columns = new_columns

        # Rename specific columns manually to match the expected names
        columns_to_rename_manually = {"elasticity_total_vehicles":'elasticity_total_vehicles_passenger', 
         "elasticity_cost_per_kilometer":'elasticity_cost_per_kilometer_passenger', 
         'elasticity_share_service_sector_gdp':'elasticity_share_service_sector_gdp_cargo_domestic', 
         'elasticity_share_elderly_65_plus':'elasticity_share_elderly_65_plus_passenger', 
         'elasticity_share_construction_sector_gdp':'elasticity_share_construction_sector_gdp_cargo_domestic', 
         'elasticity_world_trade_volume':'elasticity_world_trade_volume_cargo_international', 
         'elasticity_jobs.count.index':'elasticity_commuting_jobs_share_passenger', # Since these values are always the same
         'elasticity_higher_education_level_share':'elasticity_higher_education_level_share_passenger'}
        df = df.rename(columns=columns_to_rename_manually)

        # Set the index with experiment names
        scenario_names = generate_scenario_name_list(NUMBER_OF_SCENARIOS, core = "experiment_")
        df.index = scenario_names

        # Save to CSV
        output_path = CURR_DIR.parents[0] / "post_processing" / "surrogate_model" / "surrogate_model_inputs_paper_2.csv"
        df.to_csv(output_path)


    print("Simulation preparation completed successfully.")


if __name__ == "__main__":
    main()
