#!/usr/bin/env python3

import atexit
import json
import sys
from pathlib import Path
from tempfile import mkdtemp
import time
from movici_simulation_core import Simulation
from movici_simulation_core.core import AttributeSpec
from movici_simulation_core.core.data_type import DataType
from movici_simulation_core.models.common.attributes import CommonAttributes

CURRENT_DIR = Path(__file__).parent
INPUT_DIR = CURRENT_DIR.joinpath("data/init_data")

SCENARIO_DIR = CURRENT_DIR.joinpath('data/scenarios')
SCENARIO_NAME = "scenario_config_multimodal_test"
SCENARIO_FILE = SCENARIO_DIR.joinpath(SCENARIO_NAME).with_suffix(".json")
OUTPUT_DIR = SCENARIO_DIR.joinpath(SCENARIO_NAME)

CURRENT_DATE = time.strftime("%Y-%m-%d")
    

def run_simulation(scenario_file, output_dir):
    """
    Run the simulation with the given scenario file and output directory.
    The scenario file is a JSON file that contains the configuration of the simulation.
    The output directory is the directory where the simulation results will be stored.
    The scenario file and output directory need to be the identical, apart from the json suffix.
    """
    scenario = json.loads(Path(scenario_file).read_text())
    sim = Simulation(data_dir=INPUT_DIR, storage_dir=output_dir)
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
            AttributeSpec("transport.average_time.star", DataType(float)),
            AttributeSpec("share_service_sector_gdp", DataType(float)),
        ]
    )
    sim.configure(scenario)
    sim.run()

# Printed output will be written to a .txt file
class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # needed for Python 3 compatibility
        self.terminal.flush()
        self.log.flush()


# Define a cleanup function to reset sys.stdout to its original state
def cleanup():
    print("Cleaning up...")
    sys.stdout = sys.__stdout__  # Reset stdout to its original state


# Register the cleanup function to be called at exit
atexit.register(cleanup)

sys.stdout = DualOutput(f"{SCENARIO_NAME}_started_at_{CURRENT_DATE}.txt")

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # scenario_file = args[0]
    # output_dir = args[1]

    scenario_file = "data/scenarios/scenario_config_multimodal_test.json"
    output_dir = "data/scenarios/scenario_config_multimodal_test"

    # 127, 521 and 857 were incomplete
    run_simulation(scenario_file, output_dir)


if __name__ == "__main__":
    main()
