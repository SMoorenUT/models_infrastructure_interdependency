from pathlib import Path
from movici_simulation_core.core.schema import AttributeSchema
from movici_simulation_core.core.moment import TimelineInfo, string_to_datetime
from movici_simulation_core.postprocessing.results import SimulationResults
from movici_simulation_core.core import AttributeSpec, DataType
from movici_simulation_core.core.utils import configure_global_plugins
from movici_simulation_core.attributes import GlobalAttributes
from movici_simulation_core.models.common.attributes import CommonAttributes
from tqdm import tqdm
import datetime as dt
import numpy as np
import pandas as pd
import os

ATTRIBUTE = "transport.volume_to_capacity_ratio"
entity_number = 885  # for analysing a certain bridge.
TIMESTAMP = "2050"
DATA_TO_ANALYSE = "road_segments: delay"
BASE_DIR = Path(__file__).parents[1]
INIT_DATA_DIR = BASE_DIR / "data/init_data/"
UPDATES_DIR = Path("/media/p-drive/ET/CME/Current/Sander Mooren/Paper_2_data/output")
SIMULATION_NAME = "ema_road_model_17_07_2025"
SAVE_CSV = True  # Set to False if you do not want to save the csv files

bridge_ids_to_analyse = [671, 672, 783, 784, 1001, 785, 786, 787, 823, 824, 825, 826]
road_segment_ids_to_analyse = [
    # 554,
    # 551,
    # 3227,
    # 3228,
    # 613,
    654,
    3260,
    651,
    660,
    422,
    127,
    990,
    3299,
]

scenarios = []
for i in range(1000):
    scenario = f"{SIMULATION_NAME}_experiment_{str(i).zfill(3)}"
    scenarios.append(scenario)

# Keep `scenarios` as a list containing the single selected scenario (index 13).
# This prevents iterating the characters of a string when using `for scenario in scenarios:`.
# Drop only scenario 857 and keep all others
scenarios = [s for s in scenarios if not s.endswith(f"_experiment_{str(857).zfill(3)}")]

# Dataset configuration
if DATA_TO_ANALYSE == "bridges":
    dataset_name = "bridges"
    attribute = "transport.volume_to_capacity_ratio"
    entity_group = "bridge_entities"
    output_subdir = "bridges/individual"
    output_filename = (
        f"{SIMULATION_NAME}_transport.volume_to_capacity_ratio_{TIMESTAMP}.csv"
    )
elif DATA_TO_ANALYSE.startswith("road_network:"):
    metric = DATA_TO_ANALYSE.split(":", 1)[1].strip()
    dataset_name = "road_network"
    entity_group = "virtual_node_entities"
    output_subdir = "road_network/global"

    mapping = {  # metric: (attribute, output_filename)
        "passenger_demand_vkm": (
            "transport.passenger_demand_vkm",
            "passenger_vkt.csv",
        ),
        "passenger_demand.peak_yearly": (
            "transport.passenger_demand_vkm.peak_yearly",
            "VKT_peak_yearly.csv",
        ),
        "cargo_demand_vkm": (
            "transport.cargo_demand_vkm",
            "cargo_vkt.csv",
        ),
        "cargo_demand": ("transport.cargo_demand", "cargo_demand.csv"),
        "passenger_demand": (
            "transport.passenger_demand",
            "passenger_demand.csv",
        ),
    }
elif DATA_TO_ANALYSE.startswith("road_segments:"):
    metric = DATA_TO_ANALYSE.split(":", 1)[1].strip()
    dataset_name = "road_network"
    entity_group = "road_segment_entities"
    output_subdir = "road_network/road_segments"

    mapping = {  # metric: (attribute, output_filename)
        "delay": ("transport.delay_factor", "delay_factor.csv"),
        "passenger_vehicle_flow": (
            "transport.passenger_vehicle_flow",
            "passenger_vehicle_flow.csv",
        ),
        "cargo_vehicle_flow": (
            "transport.cargo_vehicle_flow",
            "cargo_vehicle_flow.csv",
        ),
        "passenger_car_unit": (
            "transport.passenger_car_unit",
            "passenger_car_unit.csv",
        ),
    }

    try:
        attribute, output_filename = mapping[metric]
    except KeyError:
        raise ValueError(f"Unknown road_network metric '{metric}' in DATA_TO_ANALYSE")
else:
    raise ValueError(f"Unknown DATA_TO_ANALYSE '{DATA_TO_ANALYSE}'")

OUTPUT_DIR = BASE_DIR / f"output_simulations/{SIMULATION_NAME}/{output_subdir}"

if not OUTPUT_DIR.exists():
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory {OUTPUT_DIR}")

ATTRIBUTES = [
    AttributeSpec("jobs.count.index", DataType(float)),
    AttributeSpec("people.count.index", DataType(float)),
    AttributeSpec("jobs.count", DataType(float)),
    AttributeSpec("jobs.per.capita", DataType(float)),
    AttributeSpec("jobs.density", DataType(float)),
    AttributeSpec("people.count", DataType(float)),
    AttributeSpec("people.density", DataType(float)),
    AttributeSpec("transport.international_cargo_demand", DataType(float, csr=True)),
    AttributeSpec("transport.domestic_cargo_demand", DataType(float, csr=True)),
    AttributeSpec("transport.passenger_demand", DataType(float, csr=True)),
    AttributeSpec("transport.cargo_demand_vkm", DataType(float, csr=True)),
    AttributeSpec("transport.cargo_demand_vkm.peak_yearly", DataType(float, csr=True)),
    AttributeSpec("transport.passenger_demand_vkm", DataType(float, csr=True)),
    AttributeSpec(
        "transport.passenger_demand_vkm.peak_yearly", DataType(float, csr=True)
    ),
    AttributeSpec("transport.shortest_path_length", DataType(float, csr=True)),
    AttributeSpec("transport.shortest_path_lane_length", DataType(float, csr=True)),
    AttributeSpec(
        "connection.upper_references",
        DataType(str, csr=True),
    ),
    AttributeSpec(
        "connection.lower_references",
        DataType(str, csr=True),
    ),
    AttributeSpec("transport.capacity_utilization", DataType(float)),
    AttributeSpec("transport.capacity_utilization_upper", DataType(float)),
    AttributeSpec("transport.capacity_utilization_lower", DataType(float)),
    AttributeSpec(
        "transport.automatic_incident_detection_upper",
        DataType(int),
        enum_name="kpi_status",
    ),
    AttributeSpec(
        "transport.automatic_incident_detection_lower",
        DataType(int),
        enum_name="kpi_status",
    ),
    AttributeSpec("transport.lighting_upper", DataType(int), enum_name="kpi_status"),
    AttributeSpec("transport.lighting_lower", DataType(int), enum_name="kpi_status"),
    AttributeSpec("transport.automatic_incident_detection_presence", DataType(bool)),
    AttributeSpec(
        "transport.automatic_incident_detection", DataType(int), enum_name="kpi_status"
    ),
    AttributeSpec("transport.light_presence", DataType(bool)),
    AttributeSpec("transport.lighting", DataType(int), enum_name="kpi_status"),
    AttributeSpec("noise.level", DataType(float)),
]

timeline_info = TimelineInfo(
    reference=string_to_datetime("2019").timestamp(), time_scale=1, start_time=0
)
schema = AttributeSchema(ATTRIBUTES)
schema.use(GlobalAttributes)
schema.use(CommonAttributes)
configure_global_plugins(schema)


def jan1_conversion(slice: dict) -> dict:
    """
    Converts the timestamps in the provided slice to include only the first of January for each year.

    Args:
        slice (dict): The slice of the dataset to be converted.

    Returns:
       modified_slice (dict): The modified slice containing only the first of January for each year.
    """
    dates = [timeline_info.timestamp_to_datetime(t) for t in slice["timestamps"]]
    dates_jan1_check = []
    for i in np.arange(2019, 2051):
        index = dates.index(dt.datetime(year=i, month=1, day=1))
        dates_jan1_check.append(index)
    dates = [dates[x] for x in dates_jan1_check]
    dates = [dt.datetime.strftime(x, "%Y") for x in dates]
    slice["timestamps"] = dates
    slice["data"] = [slice["data"][x] for x in dates_jan1_check]

    # for i in range(32):
    #     if "data" in slice["data"][i]:
    #         slice["data"][i] = slice["data"][i]["data"]
    return slice


def load_results(scenario, dataset_name):
    SCENARIO_UPDATES_DIR = UPDATES_DIR / scenario
    results = SimulationResults(
        INIT_DATA_DIR,
        SCENARIO_UPDATES_DIR,
        timeline_info=timeline_info,
        attributes=schema,
    )
    dataset = results.get_dataset(dataset_name)
    return dataset


def results_by_attribute(attribute, entity_group, dataset_name, save_csvs=False):
    """
    Aggregate and (optionally) persist time-series results for a specific attribute across scenarios.
    This function loads result datasets for each scenario and extracts a time-series slice
    for the provided entity group and attribute. It supports two analysis modes controlled
    by the global DATA_TO_ANALYSE value:
    - If "road_network" appears in DATA_TO_ANALYSE: sums list-of-lists values for each
        timestamp, constructs a DataFrame whose columns are scenarios and whose index are
        timestamps, and optionally writes a single CSV file containing the aggregated
        attribute timeseries for all scenarios.
    - If DATA_TO_ANALYSE == "bridges": extracts per-timestamp lists for each scenario,
        constructs a DataFrame per scenario and (optionally) writes one CSV file per
        scenario containing the bridges attribute time series.
    - If DATA_TO_ANALYSE starts with "road_segments:": extracts per-timestamp lists for each scenario,
        constructs a DataFrame per scenario and (optionally) writes one CSV file per
        scenario containing the road segments attribute time series.
    Parameters
    ----------
    attribute : str
            The attribute name to slice from each dataset (e.g., a metric or variable name).
    entity_group : str
            The entity group to slice (e.g., a modelled grouping such as a road or bridge group).
    dataset_name : str
            Name or identifier of the dataset to load for each scenario.
    save_csvs : bool, optional
            If True, write CSV files to OUTPUT_DIR. Default is False.
    Returns
    -------
    pandas.DataFrame or None
            - If DATA_TO_ANALYSE contains "road_network": returns a pandas.DataFrame where
                each column is a scenario and each index entry is a timestamp. This DataFrame
                is also written to disk as OUTPUT_DIR/output_filename if save_csvs is True.
            - If DATA_TO_ANALYSE == "bridges": the function writes one CSV per scenario
                (if save_csvs is True) and does not return a consolidated DataFrame (returns None).
            - If DATA_TO_ANALYSE starts with "road_segments:": the function writes one CSV per scenario
                (if save_csvs is True) and does not return a consolidated DataFrame (returns None).
    Raises
    ------
    Exception
            - If globals or required inputs are missing, behavior is undefined and an exception
                may be raised.
    Side effects
    ------------
    - Reads global variables such as DATA_TO_ANALYSE, scenarios, OUTPUT_DIR, and output_filename.
    - Calls external helper functions: load_results(...) and jan1_conversion(...).
    - May write CSV files to disk when save_csvs is True.
    Notes
    -----
    - The function assumes that slice["data"] may contain nested lists for road_network
        and therefore sums inner lists into a scalar per timestamp.
    - For bridges, slice["data"] entries are converted to their underlying data lists.
    - No explicit error handling for missing globals, invalid dataset slices, or I/O errors
        is performed; callers should handle such exceptions as appropriate.
    Examples
    --------
    # Aggregate road network attribute across scenarios and get a DataFrame
    df = results_by_attribute("flow", "highway", "traffic_results", save_csvs=False)
    # Aggregate bridges attribute and write CSV files for each scenario
    results_by_attribute("IC_ratio", "bridges", "infrastructure_results", save_csvs=True)
    """
    if "road_network" in DATA_TO_ANALYSE:
        road_network_attribute_dict = (
            {}
        )  # Dictionary to store the attribute data for each scenario

        for scenario in tqdm(scenarios):
            dataset = load_results(scenario, dataset_name)
            slice = dataset.slice(entity_group=entity_group, attribute=attribute)
            slice = jan1_conversion(slice)

            # A list of lists should be summed
            for i in range(len(slice["data"])):
                slice["data"][i] = np.sum(slice["data"][i]["data"])

            # Convert slice to a dictionary
            data_dict = dict(zip(slice["timestamps"], slice["data"]))

            road_network_attribute_dict[scenario] = data_dict
        attribute_df = pd.DataFrame.from_dict(road_network_attribute_dict)
        (
            attribute_df.to_csv(f"{OUTPUT_DIR}/{output_filename}", index=True)
            if save_csvs
            else None
        )

    elif DATA_TO_ANALYSE == "bridges":
        for scenario in tqdm(scenarios, desc="Bridges", colour="green"):
            dataset = load_results(scenario, dataset_name)
            slice = dataset.slice(entity_group=entity_group, attribute=attribute)
            slice = jan1_conversion(slice)

            bridges_dict = dict(zip(slice["timestamps"], slice["data"]))
            for _, year in enumerate(bridges_dict):
                # Convert to list of values
                bridges_dict[year] = bridges_dict[year]["data"]
            ic_bridges_df = pd.DataFrame.from_dict(bridges_dict)
            (
                ic_bridges_df.to_csv(
                    f"{OUTPUT_DIR}/bridges_ICratio_{scenario}.csv", index=True
                )
                if save_csvs
                else None
            )
    elif DATA_TO_ANALYSE.startswith("road_segments:"):
        for scenario in tqdm(scenarios, desc="Road Segments", colour="green"):
            dataset = load_results(scenario, dataset_name)
            slice = dataset.slice(entity_group=entity_group, attribute=attribute)
            slice = jan1_conversion(slice=slice)

            road_segments_dict = dict(zip(slice["timestamps"], slice["data"]))
            for _, year in enumerate(road_segments_dict):
                # Convert to list of values
                road_segments_dict[year] = road_segments_dict[year]["data"]
            attribute_df = pd.DataFrame.from_dict(road_segments_dict)
            (
                attribute_df.to_csv(
                    f"{OUTPUT_DIR}/road_segments_{metric}_{scenario}.csv", index=True
                )
                if save_csvs
                else None
            )


def results_by_entity(entity_number, attribute, save_csv=True):
    """
    Creates a dataframe containing the specified attribute of single bridge/road segment per scenario (rows) per year (columns). The data is saved as a csv file in the output directory.

    Args:
        entity_number (int): The entity number of the bridge/road segment to be analysed.
        attribute (str): The attribute to be analysed.
        save_csv (bool, optional): Whether to save the results as a csv file. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe containing the data for the specified entity and attribute. Rows are the scenarios, columns are the years.

    """
    attribute_df = pd.DataFrame(columns=[str(year) for year in range(2019, 2051)])
    display_group = entity_group.split("_entities")[0]
    for scenario in tqdm(
        scenarios, desc=f"{display_group} {entity_number} {attribute}", colour="green"
    ):
        dataset = load_results(scenario, dataset_name)
        slice = dataset.slice(entity_group, entity_selector=entity_number)
        slice["data"] = slice["data"][attribute]
        slice = jan1_conversion(slice=slice)

        scenario_name = scenario.split("_")[-2] + "_" + scenario.split("_")[-1]
        attribute_df.loc[scenario_name] = slice["data"]

    if save_csv:
        csv_name = f"{OUTPUT_DIR}/{SIMULATION_NAME}_RoadSegment_{entity_number}_{attribute}.csv"
        if os.path.exists(csv_name):
            user_input = input(
                f"File {csv_name} already exists. Do you want to overwrite it (o), save as a new file (n), or skip save (s)? "
            ).lower()
            if user_input == "o":
                attribute_df.to_csv(csv_name, index=True)
            elif user_input == "n":
                counter = 1
                new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                while os.path.exists(new_csv_name):
                    counter += 1
                    new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                attribute_df.to_csv(new_csv_name, index=True)
        else:
            attribute_df.to_csv(csv_name, index=True)

    return attribute_df


def results_by_attribute_and_year(
    attribute: str, timestamp: str, dataset_name="bridges", save_csv=True
) -> pd.DataFrame:
    """
    Slices the dataset over the specified attribute and timestamp. The data is saved as a csv file in the output directory.

    Args:
        attribute (str): property to be analysed as defined in simulation (e.g. "transport.volume_to_capacity_ratio")
        timestamp (str): year to be analysed (e.g. "2050")
        dataset_name (str, optional): dataset to be analysed. Defaults to "bridges".

    Returns:
        pd.DataFrame: Dataframe containing the data for the specified attribute and timestamp (i.e. year). Rows are the entities, columns are the scenarios.
    """
    data = {}
    for scenario in tqdm(scenarios):
        # Load the dataset
        dataset = load_results(scenario, dataset_name)

        # Slice the dataset over the attribute
        slice = dataset.slice(
            "bridge_entities",
            attribute=attribute,
        )
        slice = jan1_conversion(slice=slice)

        # Slice the dataset over the timestamp
        timestamp_index = slice["timestamps"].index(timestamp)
        slice["timestamps"] = slice["timestamps"][timestamp_index]
        slice["data"] = slice["data"][timestamp_index]

        # # Slice the dataset over the entity number
        # entity_index = np.where(slice["id"] == entity_number)[0][0]
        # slice["data"] = slice["data"][entity_index]
        # slice["id"] = entity_index
        scenario_name = scenario.split("_")[-2] + "_" + scenario.split("_")[-1]
        data[scenario_name] = slice["data"]["data"]
    data = pd.DataFrame.from_dict(data)

    if save_csv:
        csv_name = f"{OUTPUT_DIR}/{SIMULATION_NAME}_bridge_{entity_number}_{attribute}_{timestamp}.csv"
        if os.path.exists(csv_name):
            while True:
                user_input = input(
                    f"File {csv_name} already exists. Do you want to overwrite it (o), save as a new file (n), or skip save (s)? "
                )
                if user_input.lower() == "o":
                    data.to_csv(csv_name, index=True)
                    break
                elif user_input.lower() == "n":
                    counter = 1
                    new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                    while os.path.exists(new_csv_name):
                        counter += 1
                        new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                    data.to_csv(new_csv_name, index=True)
                    break
                elif user_input.lower() == "s":
                    break
                else:
                    print(
                        "Invalid input. Please enter 'o' to overwrite, 'n' to save as a new file, or 's' to skip save."
                    )
        else:
            data.to_csv(csv_name, index=True)

    return data


def results_by_entity_and_attribute():
    pass


def main():
    # def process_bridge_entities_from_csv():
    #     data = pd.read_csv(
    #         BASE_DIR
    #         / "output_simulations"
    #         / SIMULATION_NAME
    #         / "bridges"
    #         / f"{SIMULATION_NAME}_transport.volume_to_capacity_ratio_2050_sorted.csv",
    #         index_col=0,
    #     )
    #     entity_numbers = data.index

    for entity_number in road_segment_ids_to_analyse:
        csv_name = (
            OUTPUT_DIR
            / f"{SIMULATION_NAME}_RoadSegment_{entity_number}_{attribute}.csv"
        )
        if os.path.exists(csv_name):
            print(f"File {csv_name} already exists. Skipping...")
            continue
        results_by_entity(entity_number, attribute, save_csv=True)

    # process_bridge_entities_from_csv()

    # results_by_attribute_and_year(
    #     attribute=ATTRIBUTE,
    #     timestamp=TIMESTAMP,
    #     dataset_name="bridges",
    #     save_csv=True,
    # )

    # results_by_attribute(
    #     attribute=attribute,
    #     entity_group=entity_group,
    #     dataset_name=dataset_name,
    #     save_csvs=SAVE_CSV,
    # )


if __name__ == "__main__":
    main()
