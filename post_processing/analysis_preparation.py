import csv
import pandas as pd
from pathlib import Path
import json


# Initialize the data directories
CURR_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parents[1]
SIM_NAME = "ema_road_model_17_07_2025"
SIM_INPUT_DIR = (
    BASE_DIR / "data/init_data"
)  # Directory with the input data for the simulation
SIM_OUTPUT_DIR = (
    BASE_DIR / f"output_simulations/{SIM_NAME}"
)  # Directory with the results of the simulations
OUTPUT_DIR = BASE_DIR / "analysis"  # Directory to save the analysis data
NUMBER_OF_SCENARIOS = 1000
SIMULATION_YEARS = list(range(2019, 2051))
SAVE_CSV = True

DATA_FILES_GLOBAL = [
    "cargo_vkt.csv",
    "passenger_vkt.csv",
    "combined_vkt.csv",
    "cargo_demand.csv",
    "passenger_demand.csv",
    "combined_demand.csv",
]
MODEL_NAMES_SHORT = {4: "passenger", 5: "cargo_domestic", 6: "cargo_international"}
DATA_FILES_ROAD_SEGMENTS = [
    f.name for f in (SIM_OUTPUT_DIR / "road_network" / "road_segments").glob("*.csv")
]
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


def generate_scenario_name_list(
    number_of_scenarios, prefix=None, suffix=None, core="scenario_"
):
    len_num_samples = establish_length_num_samples(number_of_scenarios)
    scenario_name_list = [
        f"{prefix if prefix else ''}{core}{str(i).zfill(len_num_samples)}{suffix if suffix else ''}"
        for i in range(number_of_scenarios)
    ]
    return scenario_name_list


def list_of_scenarios(number_of_scenarios, suffix=None, prefix=None, core="scenario_"):
    scenarios_filenames = generate_scenario_name_list(
        number_of_scenarios, prefix=prefix, suffix=suffix, core=core
    )

    return scenarios_filenames


def load_local_data():
    scenario_filenames = (
        list_of_scenarios(
            NUMBER_OF_SCENARIOS // 2, suffix="_local_parameters_tape.json"
        )
        * 2
    )  # Each local parameters file is used twice in the simulation
    local_parameter_files = [
        json.load(open(SIM_INPUT_DIR / file)) for file in scenario_filenames
    ]
    return local_parameter_files


def load_global_data():
    scenario_filenames = (
        list_of_scenarios(NUMBER_OF_SCENARIOS // 2, suffix="_global_parameters.csv") * 2
    )  # Each global parameters file is used twice in the simulation
    # Drop the scenario (857) with no data
    scenario_filenames.pop(857)
    global_parameter_files = {
        file: pd.read_csv(SIM_INPUT_DIR / file) for file in scenario_filenames
    }
    return global_parameter_files


def load_elasticity_data():
    scenarios_dir = SIM_INPUT_DIR.parent / "scenarios"
    scenario_filenames = list_of_scenarios(
        NUMBER_OF_SCENARIOS, prefix=SIM_NAME, suffix=".json", core="_experiment_"
    )
    elasticity_data = {}
    for file in scenario_filenames:
        with open(scenarios_dir / file, "r") as f:
            elasticity_data[file] = json.load(f)
    return elasticity_data


def load_data():
    local_data = load_local_data()
    global_data = load_global_data()
    elasticity_data = load_elasticity_data()
    return (
        local_data,
        global_data,
        elasticity_data,
    )  # Return a tuple with the local, global, and elasticity data


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
    half = len(scenario_name_list) // 2
    for idx, scenario_name in enumerate(scenario_name_list):
        # use only the first half of scenario names (repeated) to pick CSV files
        first_half_name = scenario_name_list[idx % half]
        csv = global_data[f"{first_half_name}_global_parameters.csv"]
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


def reshape_elasticity_data(elasticity_data, scenario_name_list) -> pd.DataFrame:
    experiment_name_list = list_of_scenarios(
        NUMBER_OF_SCENARIOS, prefix=SIM_NAME, suffix=".json", core="_experiment_"
    )
    elasticity_data_dict = {}
    for idx, scenario_name in enumerate(scenario_name_list):
        exp_name = experiment_name_list[idx]
        exp = elasticity_data.get(exp_name, {})
        models = exp.get("models", {})
        # Keep only models with type == "traffic_demand_calculation"
        if isinstance(models, dict):
            models = {
                k: v
                for k, v in models.items()
                if isinstance(v, dict) and v.get("type") == "traffic_demand_calculation"
            }
        else:
            # handle list-like models: use 'id' if present, otherwise the index as key
            filtered = {}
            for idx, m in enumerate(models or []):
                if (
                    isinstance(m, dict)
                    and m.get("type") == "traffic_demand_calculation"
                ):
                    key = m.get("id", idx)
                    filtered[key] = m
                models = filtered

        gp = {}  # Initialize an empty dict to collect global parameters
        # For each model 4,5,6, extract its global_parameters dict and store values
        for model in (4, 5, 6):
            # accept both int and str keys for the models mapping
            model_key = model if model in models else str(model)
            model_entry = models[model_key]
            gp_current_model = model_entry.get("global_parameters", {})
            if isinstance(gp_current_model, list):
                transformed = {}
                for item in gp_current_model:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if name is None:
                        continue
                    transformed[name] = item.get("elasticity")
                gp_current_model = transformed
            elif isinstance(gp_current_model, dict):
                # If values are dicts that contain 'name'/'elasticity', convert them to name: elasticity
                if any(
                    isinstance(v, dict) and ("name" in v or "elasticity" in v)
                    for v in gp_current_model.values()
                ):
                    transformed = {}
                    for k, v in gp_current_model.items():
                        if isinstance(v, dict) and "name" in v:
                            transformed[v["name"]] = v.get("elasticity")
                        else:
                            transformed[k] = v
                    gp_current_model = transformed
            # Rename the keys to indicate the model they belong to
            gp_current_model = {
                f"elasticity_{key}_{MODEL_NAMES_SHORT[model]}": value
                for key, value in gp_current_model.items()
            }
            # append the gp_current_model dict to a list under the 'models' key
            gp.update(gp_current_model)
            pass
        pass

        elasticity_data_dict[scenario_name] = (
            gp  # Overwrite with the processed global_parameters dict
        )

    # Turn dict into a dataframe
    elasticity_data_df = pd.DataFrame(elasticity_data_dict).T

    return elasticity_data_df


def process_init_data(
    local_data, global_data, elasticity_data, year=2050
) -> pd.DataFrame:
    # Initialize values for later use
    number_of_scenarios = len(local_data)
    scenario_name_list = generate_scenario_name_list(
        number_of_scenarios=number_of_scenarios
    )
    number_of_timesteps = len(local_data[0]["data"]["time_series"])
    year_position = SIMULATION_YEARS.index(year)

    # Reshape the data into pandas dataframes
    # First the local data
    local_data_df = reshape_local_data(
        local_data, scenario_name_list, year_position=year_position
    )
    # Second the global data
    global_data_df = reshape_global_data(
        global_data, scenario_name_list, year_position=year_position
    )
    # Third the elasticity data
    elasticity_data_df = reshape_elasticity_data(elasticity_data, scenario_name_list)

    # Ignore local data since it is derived from the global population and jobs data.
    combined_data_df = pd.concat([global_data_df, elasticity_data_df], axis=1)

    # Rename the index from scenario's to experiment's
    combined_data_df.index = combined_data_df.index.str.replace(
        "scenario_", "experiment_"
    )

    # Drop the row with index "experiment_857" if it exists
    combined_data_df = combined_data_df.drop(index="experiment_857", errors="ignore")

    n = len(combined_data_df)
    combined_data_df["blankenburgverbinding"] = [1] * min(500, n) + [0] * max(
        0, n - 500
    )

    return combined_data_df


def load_output_data(filepath, transpose=False) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    if transpose:
        df = df.T
    return df


def process_output_data(data_files_global, data_files_road_segments, year=2050):
    SIM_OUTPUT_DIR_TEMP = SIM_OUTPUT_DIR / "road_network"

    data_files_dfs = {}
    data = {}

    def _get_row_values_for_year(df, year, file_name):
        # Try direct label lookup first
        if year in df.index:
            row = df.loc[year]
        else:
            # Try numeric match if index contains numeric-like labels
            idx_numeric = pd.to_numeric(df.index, errors="coerce")
            matches = idx_numeric == year
            if matches.any():
                # pick the first matching row
                row = df.loc[matches].iloc[0]
            else:
                # Helpful error with examples of available indices
                sample_idx = list(df.index[:10])
                raise IndexError(
                    f"Year {year} not found in file '{file_name}'. "
                    f"Available index samples: {sample_idx} (total {len(df.index)} rows)."
                )
        # row may be a Series or DataFrame -> make a flat list
        return row.values.tolist() if hasattr(row, "values") else list(row)

    # Process global data files
    for file in data_files_global:
        data_files_dfs[file] = load_output_data(SIM_OUTPUT_DIR_TEMP / "global" / file)
        name = Path(file).stem
        data[name + f"_year_{year}"] = _get_row_values_for_year(
            data_files_dfs[file], year, file
        )

    # Process road segment data files
    for file in data_files_road_segments:
        data_files_dfs[file] = load_output_data(
            SIM_OUTPUT_DIR_TEMP / "road_segments" / file, transpose=True
        )
        stem = Path(file).stem
        prefix = f"{SIM_NAME}_"
        name = stem[len(prefix) :] if stem.startswith(prefix) else stem
        data[name + f"_year_{year}"] = _get_row_values_for_year(
            data_files_dfs[file], year, file
        )

    output_data_df = pd.DataFrame(data)
    scenario_names = generate_scenario_name_list(
        len(output_data_df) + 1, core="experiment_"
    )
    scenario_names.pop(
        857
    )  # removed is the popped string; scenario_names now lacks that element
    output_data_df.index = scenario_names

    return output_data_df


def main(save_to_csv=False):

    local_data, global_data, elasticity_data = load_data()
    init_data_df = process_init_data(
        local_data, global_data, elasticity_data, year=year
    )
    output_data_df = process_output_data(
        DATA_FILES_GLOBAL, DATA_FILES_ROAD_SEGMENTS, year=year
    )

    analysis_ready_df = pd.concat([init_data_df, output_data_df], axis=1)

    (
        analysis_ready_df.to_csv(OUTPUT_DIR / f"{SIM_NAME}_results.csv")
        if save_to_csv
        else None
    )


if __name__ == "__main__":
    main(save_to_csv=SAVE_CSV)
    print("Finished running script.")
