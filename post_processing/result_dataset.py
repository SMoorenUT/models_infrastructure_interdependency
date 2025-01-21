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

attribute = "transport.volume_to_capacity_ratio"
enitity_number = 720  # for analysing a certain bridge for example
timestamp = "2050"
DATA_TO_ANALYSE = "bridges"
BASE_DIR = Path(__file__).parents[1]
INIT_DATA_DIR = BASE_DIR / "data/init_data/"
UPDATES_DIR = Path(
    "/media/p-drive/ET/CME/Current/Sander Mooren/scenarios_ema_1000/Output/"
)
SIMULATION_NAME = "ema_road_model_08_05_2024"

scenarios = []
for i in range(1000):
    scenario = f"{SIMULATION_NAME}_scenario_{str(i).zfill(3)}"
    scenarios.append(scenario)

if DATA_TO_ANALYSE == "bridges":
    dataset_name = "bridges"
    attribute = "transport.volume_to_capacity_ratio"
    entity_group = "bridge_entities"
    output_subdir = "bridges/individual"
    output_filename = "ICratio_{enitity_number}.csv"
elif DATA_TO_ANALYSE == "road_network: passenger_demand_vkm":
    dataset_name = "road_network"
    attribute = "transport.passenger_demand_vkm"
    entity_group = "virtual_node_entities"
    output_subdir = "road_network"
    output_filename = "passenger_vkm.csv"
elif DATA_TO_ANALYSE == "road_network: passenger_demand.peak_yearly":
    dataset_name = "road_network"
    attribute = "transport.passenger_demand.peak_yearly"
    entity_group = "virtual_node_entities"
    output_subdir = "road_network"
    output_filename = "VKM_peak_yearly.csv"
elif DATA_TO_ANALYSE == "road_network: cargo_demand_vkm":
    dataset_name = "road_network"
    attribute = "transport.cargo_demand_vkm"
    entity_group = "virtual_node_entities"
    output_subdir = "road_network"
    output_filename = "cargo_vkm.csv"
else:
    print("Dataset not found")

OUTPUT_DIR = BASE_DIR / f"output_simulations/ema_road_model_08_05_2024/{output_subdir}"

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
    if "road_network" in DATA_TO_ANALYSE:
        road_network_vkm_dict = {}  # Dictionary to store the vkm data for each scenario

        for scenario in tqdm(scenarios):
            dataset = load_results(scenario, dataset_name)
            slice = dataset.slice(entity_group=entity_group, attribute=attribute)
            slice = jan1_conversion(slice, entity_group, attribute)

            # Convert slice to a dictionary
            data_dict = dict(zip(slice["timestamps"], slice["data"]))

            # VKM is originally a list of lists, should be summed.
            for key in data_dict:
                data_dict[key] = np.sum(data_dict[key])

            road_network_vkm_dict[scenario] = data_dict
        vkm_df = pd.DataFrame.from_dict(road_network_vkm_dict)
        (
            vkm_df.to_csv(f"{OUTPUT_DIR}/{output_filename}", index=True)
            if save_csvs
            else None
        )

    elif DATA_TO_ANALYSE == "bridges":
        for scenario in scenarios:
            dataset = load_results(scenario, dataset_name)
            slice = dataset.slice(entity_group=entity_group, attribute=attribute)
            slice = jan1_conversion(slice, entity_group, attribute)

            bridges_dict = dict(zip(slice["timestamps"], slice["data"]))
            ic_bridges_df = pd.DataFrame.from_dict(bridges_dict)
            (
                ic_bridges_df.to_csv(
                    f"{OUTPUT_DIR}/bridges_ICratio_{scenario}.csv", index=True
                )
                if save_csvs
                else None
            )


def results_by_entity(entity_number, attribute, save_csv=True):
    """
    Creates a dataframe containing the specified attribute of single bridge per scenario (rows) per year (columns). The data is saved as a csv file in the output directory.

    Args:
        entity_number (int): The entity number to be analysed.
        attribute (str): The attribute to be analysed.
        save_csv (bool, optional): Whether to save the results as a csv file. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe containing the data for the specified entity and attribute. Rows are the scenarios, columns are the years.

    """
    ic_bridges_df = pd.DataFrame(columns=[str(year) for year in range(2019, 2051)])
    for scenario in tqdm(scenarios, desc=f"Bridge {entity_number}", colour="green"):
        dataset = load_results(scenario, dataset_name)
        slice = dataset.slice("bridge_entities", entity_selector=entity_number)
        slice["data"] = slice["data"][attribute]
        slice = jan1_conversion(slice=slice)

        scenario_name = scenario.split("_")[-2] + "_" + scenario.split("_")[-1]
        ic_bridges_df.loc[scenario_name] = slice["data"]

    if save_csv:
        csv_name = f"{OUTPUT_DIR}/{SIMULATION_NAME}_Bridge_{entity_number}_ICratio.csv"
        if os.path.exists(csv_name):
            user_input = input(
                f"File {csv_name} already exists. Do you want to overwrite it (o), save as a new file (n), or skip save (s)? "
            ).lower()
            if user_input == "o":
                ic_bridges_df.to_csv(csv_name, index=True)
            elif user_input == "n":
                counter = 1
                new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                while os.path.exists(new_csv_name):
                    counter += 1
                    new_csv_name = f"{csv_name.split('.csv')[0]}_{counter}.csv"
                ic_bridges_df.to_csv(new_csv_name, index=True)
        else:
            ic_bridges_df.to_csv(csv_name, index=True)

    return ic_bridges_df


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
        pd.DataFrame: Dataframe containing the data for the specified attribute and timestamp. Rows are the entities, columns are the scenarios.
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
        data[scenario_name] = slice["data"]
    data = pd.DataFrame.from_dict(data)

    if save_csv:
        csv_name = f"{OUTPUT_DIR}/{SIMULATION_NAME}/bridges/{SIMULATION_NAME}_{attribute}_{timestamp}.csv"
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
    data = pd.read_csv(
        BASE_DIR
        / "output_simulations"
        / "ema_road_model_08_05_2024"
        / "bridges"
        / "ema_road_model_08_05_2024_transport.volume_to_capacity_ratio_2050_sorted.csv",
        index_col=0,
    )
    entity_numbers = data.index
    # results_by_attribute(attribute, entity_group, dataset_name, save_csvs=True)
    for entity_number in entity_numbers:
        csv_name = f"{OUTPUT_DIR}/{SIMULATION_NAME}_Bridge_{entity_number}_ICratio.csv"
        if os.path.exists(csv_name):
            print(f"File {csv_name} already exists. Skipping...")
            continue
        results_by_entity(entity_number, attribute, save_csv=True)
    # results_by_attribute_and_year(
    #     attribute=attribute,
    #     timestamp=timestamp,
    #     dataset_name="bridges",
    #     save_csv=True,
    # )


if __name__ == "__main__":
    main()
