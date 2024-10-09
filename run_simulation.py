#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from tempfile import mkdtemp
from movici_simulation_core import Simulation
from movici_simulation_core.core import AttributeSpec
from movici_simulation_core.core.data_type import DataType
from movici_simulation_core.models.common.attributes import CommonAttributes

CURRENT_DIR = Path(__file__).parent
input_dir = CURRENT_DIR.joinpath("data/init_data")

# output_dir = CURRENT_DIR.joinpath("simulations")

# SCENARIO_DIR = CURRENT_DIR.joinpath('data/scenarios')
# SCENARIO_NAME = "green"
# SCENARIO_FILE = SCENARIO_DIR.joinpath(SCENARIO_NAME).with_suffix(".json")
# SCENARIO_FILE = CURRENT_DIR.joinpath('scenario.json')

def run_simulation(scenario_file, output_dir):
    """
    Run the simulation with the given scenario file and output directory.
    The scenario file is a JSON file that contains the configuration of the simulation.
    The output directory is the directory where the simulation results will be stored.
    The scenario file and output directory need to be the identical, apart from the json suffix.
    """
    scenario = json.loads(Path(scenario_file).read_text())
    sim = Simulation(data_dir=input_dir, storage_dir=output_dir)
    sim.use(CommonAttributes)
    sim.register_attributes(
        [
            AttributeSpec("jobs.count.index", DataType(float)),
            AttributeSpec("people.count.index", DataType(float)),
            AttributeSpec("jobs.count", DataType(float)),
            AttributeSpec("jobs.per.capita", DataType(float)),
            AttributeSpec("jobs.density", DataType(float)),
            AttributeSpec("people.count", DataType(float)),
            AttributeSpec("people.density", DataType(float)),
            AttributeSpec(
                "transport.international_cargo_demand", DataType(float, csr=True)
            ),
            AttributeSpec("transport.domestic_cargo_demand", DataType(float, csr=True)),
            AttributeSpec("transport.passenger_demand", DataType(float, csr=True)),
            AttributeSpec("transport.cargo_demand_vkm", DataType(float, csr=True)),
            AttributeSpec(
                "transport.cargo_demand_vkm.peak_yearly", DataType(float, csr=True)
            ),
            AttributeSpec("transport.passenger_demand_vkm", DataType(float, csr=True)),
            AttributeSpec(
                "transport.passenger_demand_vkm.peak_yearly", DataType(float, csr=True)
            ),
            AttributeSpec("transport.shortest_path_length", DataType(float, csr=True)),
            AttributeSpec(
                "transport.shortest_path_lane_length", DataType(float, csr=True)
            ),
        ]
    )
    sim.configure(scenario)
    sim.run()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # scenario_file = args[0]
    # output_dir = args[1]

    scenario_file = (
        "data/scenarios_ema_1000/ema_road_model_08_05_2024_scenario_009.json"
    )
    output_dir = "data/scenarios_ema_1000/ema_road_model_08_05_2024_scenario_009"

    run_simulation(scenario_file, output_dir)

if __name__ == "__main__":
    main()
