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
SCENARIO_STEM = "data/scenarios/ema_road_model_27_11_2025_experiment_"
NUM_SIMULATIONS = 10
LEN_SIM = 0 # Number of digits in the simulation number
STARTING_NUMBER = 0
TASKS = [f"{SCENARIO_STEM}{i}.json" for i in range(STARTING_NUMBER, STARTING_NUMBER + NUM_SIMULATIONS)]

input_dir = Path("data/init_data")
output_dir = Path("data/scenarios")
 

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
        for number in zip(TASKS, executor.map(run_simulation, TASKS)):
            print('Completed simulation for scenario file:', number)


if __name__ == "__main__":
    main()