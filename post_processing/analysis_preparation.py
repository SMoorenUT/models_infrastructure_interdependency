import csv
import pandas as pd
from pathlib import Path
import json


# Initialize the data directories
CURR_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parents[1]
SIM_NAME = "ema_road_model_08_05_2024"
SIM_INPUT_DIR = (
    BASE_DIR / "data/init_data"
)  # Directory with the input data for the simulation
SIM_OUTPUT_DIR = (
    BASE_DIR / f"output_simulations/{SIM_NAME}"
)  # Directory with the results of the simulations
OUTPUT_DIR = BASE_DIR / "analysis"  # Directory to save the analysis data
NUMBER_OF_SCENARIOS = 1000
SIMULATION_YEARS = list(range(2019, 2051))

data_files = ["cargo_vkm.csv", "passenger_vkm.csv", "combined_vkm.csv"]
data_files = [
    "passenger_vkm.csv"
]  # Overwrite because this case only has passenger_vkm.csv
year = 2050


def establish_length_num_samples(num_samples: int):
    if num_samples <= 0:
        ValueError("num_samples must be greater than 0")
    length_num_samples = (
        len(str(num_samples))
        if num_samples not in [10**i for i in range(10)]
        else len(str(num_samples)) - 1
    )  # To format het scenario names with leading zeros
    return length_num_samples


def generate_scenario_name_list(number_of_scenarios, prefix=None, suffix=None):
    len_num_samples = establish_length_num_samples(number_of_scenarios)
    scenario_name_list = [
        f"{prefix if prefix else ''}scenario_{str(i).zfill(len_num_samples)}{suffix if suffix else ''}"
        for i in range(number_of_scenarios)
    ]
    return scenario_name_list


def list_of_scenarios(number_of_scenarios, suffix=None, prefix=None):
    scenarios_filenames = generate_scenario_name_list(
        number_of_scenarios, prefix=prefix, suffix=suffix
    )

    return scenarios_filenames


def load_local_data():
    scenario_filenames = list_of_scenarios(
        NUMBER_OF_SCENARIOS, suffix="_local_parameters_tape.json"
    )
    local_parameter_files = [
        json.load(open(SIM_INPUT_DIR / file)) for file in scenario_filenames
    ]
    return local_parameter_files


def load_global_data():
    scenario_filenames = list_of_scenarios(
        NUMBER_OF_SCENARIOS, suffix="_global_parameters.csv"
    )
    global_parameter_files = {
        file: pd.read_csv(SIM_INPUT_DIR / file) for file in scenario_filenames
    }
    return global_parameter_files


def load_data():
    local_data = load_local_data()
    global_data = load_global_data()
    return local_data, global_data  # Return a tuple with the local and global data


def reshape_local_data(
    local_data, scenario_name_list, year_position=-1
) -> pd.DataFrame:
    # Extract the data from the nested structure
    for i in range(NUMBER_OF_SCENARIOS):
        local_data[i] = local_data[i]["data"]["data_series"]
        local_data[i] = local_data[i][
            year_position
        ]  # Select the data for the last timestep by slicing over the last element

    local_data_dict = {}
    for idx, scenario_name in enumerate(scenario_name_list):
        local_data_dict[scenario_name] = local_data[idx]
        local_data_dict[scenario_name] = local_data_dict[scenario_name]["area_entities"]

    # Fill a new dataframe in the right format
    # TODO: Implement zero filling for "jobs_XX" and "population_XX" columns
    local_data_dict_formatted = {}
    # First, loop over jobs
    # TODO: Refactor first slice to be dynamic to varying scenario lengths
    for idx in range(len((local_data_dict["scenario_000"]["id"]))):
        local_data_dict_formatted[f"jobs_{idx}"] = [
            local_data_dict[scenario_name]["jobs.count.index"][idx]
            for scenario_name in scenario_name_list
        ]
    # Then loop over population
    for idx in range(len((local_data_dict["scenario_000"]["id"]))):
        local_data_dict_formatted[f"population_{idx}"] = [
            local_data_dict[scenario_name]["people.count.index"][idx]
            for scenario_name in scenario_name_list
        ]

    # Turn dict into a dataframe
    local_data_df = pd.DataFrame(local_data_dict_formatted, index=scenario_name_list)

    return local_data_df


def reshape_global_data(
    global_data, scenario_name_list, year_position=-1
) -> pd.DataFrame:
    global_data_dict = {}
    for scenario_name in scenario_name_list:
        csv = global_data[f"{scenario_name}_global_parameters.csv"]
        global_data_dict[scenario_name] = csv.to_dict(orient="records")

    global_variables = list(global_data_dict[scenario_name_list[0]][0].keys())
    global_data_dict_formatted = {}
    for variable in global_variables:
        global_data_dict_formatted[variable] = [
            global_data_dict[scenario_name][year_position][variable]
            for scenario_name in scenario_name_list
        ]

    # Turn dict into a dataframe
    global_data_df = pd.DataFrame(global_data_dict_formatted, index=scenario_name_list)
    global_data_df = global_data_df.drop(columns=["seconds", "_year"])
    return global_data_df


def process_init_data(local_data, global_data, year=2050) -> pd.DataFrame:
    # Initialize values for later use
    number_of_scenarios = len(local_data)
    scenario_name_list = generate_scenario_name_list(
        number_of_scenarios=number_of_scenarios
    )
    number_of_timesteps = len(local_data[0]["data"]["time_series"])
    year_position = SIMULATION_YEARS.index(year)

    # Reshape the data into pandas dataframes
    local_data_df = reshape_local_data(
        local_data, scenario_name_list, year_position=year_position
    )
    global_data_df = reshape_global_data(
        global_data, scenario_name_list, year_position=year_position
    )
    combined_data_df = pd.concat([global_data_df, local_data_df], axis=1)

    return combined_data_df


def load_output_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


def process_output_data(data_files, year=2050):
    SIM_OUTPUT_DIR_TEMP = SIM_OUTPUT_DIR / "road_network"

    data_files_dfs = {}
    data = {}

    for file in data_files:
        data_files_dfs[file] = load_output_data(SIM_OUTPUT_DIR_TEMP / file)
        data[file + "_year"] = data_files_dfs[file][
            data_files_dfs[file].index == year
        ].values.tolist()[0]

    output_data_df = pd.DataFrame(data)
    output_data_df.index = generate_scenario_name_list(len(output_data_df))

    return output_data_df


def main(save_to_csv=False):

    local_data, global_data = load_data()
    init_data_df = process_init_data(local_data, global_data, year=year)
    output_data_df = process_output_data(data_files, year=year)

    analysis_ready_df = pd.concat([init_data_df, output_data_df], axis=1)

    (
        analysis_ready_df.to_csv(OUTPUT_DIR / f"{SIM_NAME}_results.csv")
        if save_to_csv
        else None
    )


if __name__ == "__main__":
    main(save_to_csv=True)
    print("Finished running script.")
