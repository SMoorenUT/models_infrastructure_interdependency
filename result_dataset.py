from pathlib import Path
from movici_simulation_core.core.schema import AttributeSchema

from movici_simulation_core.utils.moment import TimelineInfo, string_to_datetime
from movici_simulation_core.postprocessing.results import SimulationResults
from movici_simulation_core.core import AttributeSpec, DataType
from movici_simulation_core.core.utils import configure_global_plugins
from movici_simulation_core.core.attributes import GlobalAttributes
from movici_simulation_core.models.common.attributes import CommonAttributes
import datetime as dt
import numpy as np
import pandas as pd

scenarios = ["infraconomy", "safety_revolution", "missed_boat", "green"]
attribute = "transport.volume_to_capacity_ratio"
# attribute = "noise.level"
# attribute = "transport.automatic_incident_detection"
# attribute = "transport.lighting"



BASE_DIR = Path(__file__).parent
INIT_DATA_DIR = BASE_DIR / "data/init_data/"
UPDATES_DIR = BASE_DIR / "data/scenarios/"

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
    AttributeSpec("transport.passenger_demand_vkm.peak_yearly", DataType(float, csr=True)),
    AttributeSpec("transport.shortest_path_length", DataType(float, csr=True)),
    AttributeSpec("transport.shortest_path_lane_length", DataType(float, csr=True)),
]
if __name__ == "__main__":
    timeline_info = TimelineInfo(
        reference=string_to_datetime("2019").timestamp(), time_scale=1, start_time=0
    )
    schema = AttributeSchema(ATTRIBUTES)
    schema.use(GlobalAttributes)
    schema.use(CommonAttributes)
    configure_global_plugins(schema)

    for scenario in scenarios:
        SCENARIO_UPDATES_DIR = UPDATES_DIR / scenario

        results = SimulationResults(INIT_DATA_DIR, SCENARIO_UPDATES_DIR, timeline_info=timeline_info, attributes=schema)
        
        dataset = results.get_dataset("bridges")
        
        slice = dataset.slice("bridge_entities", attribute=attribute)
        dates = [timeline_info.timestamp_to_datetime(t) for t in slice['timestamps']]
        
        dates_jan1_check = []
        for i in np.arange(2019, 2051):
            index = dates.index(dt.datetime(year = i, month=1, day=1))
            dates_jan1_check.append(index)
        dates = [dates[x] for x in dates_jan1_check]
        dates = [dt.datetime.strftime(x, "%Y") for x in dates]
        slice["timestamps"] = dates
        slice["data"] = [slice["data"][x] for x in dates_jan1_check]


        for i in range(32):
            slice["data"][i] = slice["data"][i]["data"]

        bridges = dict(zip(slice["timestamps"], slice["data"])) 
        ic_bridges_df = pd.DataFrame.from_dict(bridges)
        ic_bridges_df.to_csv(f"bridges_lighting_{scenario}.csv", index=True)

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