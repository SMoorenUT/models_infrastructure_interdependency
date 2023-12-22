import json
from pathlib import Path
from movici_simulation_core.preprocessing.tapefile import (
    InterpolatingTapefile,
    TimeDependentAttribute,
)

scenarios = ["infraconomy", "missed_boat", "safety_revolution", "green"]

def get_initdata(filename):
    return json.loads(Path(filename).read_text())

# given some init data
init_data = get_initdata("data/init_data/bridges.json")

for scenario in scenarios:
    # create a tapefile
    tapefile = InterpolatingTapefile(
        init_data["data"]["bridge_entities"],
        dataset_name="bridges",
        entity_group_name="bridge_entities",
        reference="reference",
        tapefile_name=f"{scenario}_noise_tapefile",
    )

    # add the csv files as TimeDependentAttribute (multiple allowed)
    tapefile.add_attribute(
        TimeDependentAttribute(name="noise.level", csv_file=f"data/noise/temp_noise_{scenario}.csv", key="bo_id")
    )

    # dump the tapefile to an output file
    tapefile.dump(file=f"{scenario}_noise_tapefile.json")