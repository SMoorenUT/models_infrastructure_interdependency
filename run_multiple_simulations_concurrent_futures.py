import run_simulation
import json
import sys
from pathlib import Path
from tempfile import mkdtemp
from movici_simulation_core import Simulation
from movici_simulation_core.core import AttributeSpec
from movici_simulation_core.core.data_type import DataType
from movici_simulation_core.models.common.attributes import CommonAttributes
import concurrent.futures
import math
import tempfile
import os

NUMBER_OF_SIMULATIONS_PARALLEL = 5
SCENARIO_STEM = "data/scenarios_ema_1000/ema_road_model_08_05_2024_scenario_"
NUM_SIMULATIONS = 1000
LEN_SIM = 3 # Number of digits in the simulation number
STARTING_NUMBER = 0

# retrieve list of all json files with os.path...
TASKS = ["./data/experiments/experiment_001.json",
         "./data/experiments/experiment_002.json"]

input_dir = Path("data/init_data")
output_dir = Path("simulations")
 

def run_simulation(experiment_file):
    scenario = json.loads(Path(experiment_file).read_text())
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

   

 

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for number, prime in zip(TASKS, executor.map(run_simulation, TASKS)):
            print('%d is prime: %s' % (number, prime))


if __name__ == "__main__":
    main()