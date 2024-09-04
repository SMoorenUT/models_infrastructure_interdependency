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

scenarios = []
enitity_number = 0  # for analysing a certain bridge for example
DATA_TO_ANALYSE = "road_network: passenger_demand_vkm"
BASE_DIR = Path(__file__).parents[1]
INIT_DATA_DIR = BASE_DIR / "data/init_data/"
UPDATES_DIR = BASE_DIR / "data/scenarios_ema_1000/"

for i in range(585):
    scenario = f"ema_road_model_08_05_2024_scenario_{str(i).zfill(3)}"
    scenarios.append(scenario)

scenarios = scenarios[-500:]

if DATA_TO_ANALYSE == "bridges":
    dataset_name = "bridges"
    attribute = "transport.volume_to_capacity_ratio"
    entity_group = "bridge_entities"
    output_subdir = "bridges"
    output_filename = "ICratio.csv"
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
]

timeline_info = TimelineInfo(
    reference=string_to_datetime("2019").timestamp(), time_scale=1, start_time=0
)
schema = AttributeSchema(ATTRIBUTES)
schema.use(GlobalAttributes)
schema.use(CommonAttributes)
configure_global_plugins(schema)


def jan1_conversion(dataset, entity_group, attribute):
    slice = dataset.slice(entity_group=entity_group, attribute=attribute)
    dates = [timeline_info.timestamp_to_datetime(t) for t in slice["timestamps"]]
    dates_jan1_check = []
    for i in np.arange(2019, 2051):
        index = dates.index(dt.datetime(year=i, month=1, day=1))
        dates_jan1_check.append(index)
    dates = [dates[x] for x in dates_jan1_check]
    dates = [dt.datetime.strftime(x, "%Y") for x in dates]
    slice["timestamps"] = dates
    slice["data"] = [slice["data"][x] for x in dates_jan1_check]

    for i in range(32):
        slice["data"][i] = slice["data"][i]["data"]
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
            slice = jan1_conversion(dataset, entity_group, attribute)

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
            bridges_dict = dict(zip(slice["timestamps"], slice["data"]))
            ic_bridges_df = pd.DataFrame.from_dict(bridges_dict)
            (
                ic_bridges_df.to_csv(
                    f"{OUTPUT_DIR}/bridges_ICratio_{scenario}.csv", index=True
                )
                if save_csvs
                else None
            )


def results_by_entity(enitity_number):
    for scenario in scenarios:
        SCENARIO_UPDATES_DIR = UPDATES_DIR / scenario

        results = SimulationResults(
            INIT_DATA_DIR,
            SCENARIO_UPDATES_DIR,
            timeline_info=timeline_info,
            attributes=schema,
        )

        dataset = results.get_dataset("bridges")

        slice = dataset.slice("bridge_entities", entity_selector=0)
        dates = [timeline_info.timestamp_to_datetime(t) for t in slice["timestamps"]]

        dates_jan1_check = []
        for i in np.arange(2019, 2051):
            index = dates.index(dt.datetime(year=i, month=1, day=1))
            dates_jan1_check.append(index)
        dates = [dates[x] for x in dates_jan1_check]
        dates = [dt.datetime.strftime(x, "%Y") for x in dates]
        slice["timestamps"] = dates
        slice["data"] = [slice["data"][x] for x in dates_jan1_check]

        for i in range(32):
            slice["data"][i] = slice["data"][i]["data"]

        bridges = dict(zip(slice["timestamps"], slice["data"]))
        ic_bridges_df = pd.DataFrame.from_dict(bridges)
        ic_bridges_df.to_csv(f"{OUTPUT_DIR}/bridges_ICratio_{scenario}.csv", index=True)


if __name__ == "__main__":
    results_by_attribute(attribute, entity_group, dataset_name, save_csvs=True)
    # results_by_entity(enitity_number)


# print(
#     "Slicing a dataset over a specific attribute",
#     slice,
#     sep="\n",
# )
# print(
#     "Slicing a dataset over a specific entity (entity ID 12)",
#     dataset.slice(dataset, entity_selector=12),
#     sep="\n",
# )
#
# print(
#     "Slicing a dataset over a specific timestamp",
#     dataset.slice(dataset, timestamp="2020"),
#     sep="\n",
# )
