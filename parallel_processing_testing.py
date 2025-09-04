import atexit
import concurrent.futures
import json
import math
from pathlib import Path
import concurrent.futures
import math
import sys
import time
from run_simulation import run_simulation
from tqdm import tqdm
import json
from pathlib import Path
from movici_simulation_core import Simulation
from movici_simulation_core.core import AttributeSpec
from movici_simulation_core.core.data_type import DataType
from movici_simulation_core.models.common.attributes import CommonAttributes
from datetime import datetime

CURRENT_DIR = Path(__file__).parent
input_dir = CURRENT_DIR.joinpath("data/init_data")
stem = "data/scenarios/ema_road_model_21_05_2025_scenario_"
scenarios = [f"{stem}{i}.json" for i in range(10)]

simulation_name = stem.split("/")[-1].split("_scenario_", 1)[0]
logfile = open(f"{simulation_name}_log.txt", "a")


def run_simulation_single_arg(scenario_file):
    """
    Run the simulation with the given scenario file and output directory.
    The scenario file is a JSON file that contains the configuration of the simulation.
    The output directory is the directory where the simulation results will be stored.
    The scenario file and output directory need to be the identical, apart from the json suffix.
    """
    scenario = json.loads(Path(scenario_file).read_text())
    output_dir = scenario_file.removesuffix(".json")
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
    # Print when a scenario started
    scenario_name = scenario_file.split("/")[-1]
    simulation_name = scenario_name.split("_scenario_", 1)[0]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Started {scenario_file} for {simulation_name}.", file=logfile)

    sim.configure(scenario)
    sim.run()

    # Print when a scenario is finished

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Finished {scenario_file} for {simulation_name}.", file=logfile)


# starting_number = int(100000)
# growth_factor = 4
# length_of_simulation = 1000
# NUMBERS = list(
#     range(
#         int(starting_number * growth_factor), int(starting_number * growth_factor + 10)
#     )
# )


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def timemit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@timemit
def run_concurrent():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for scenario in tqdm(
            executor.map(run_simulation_single_arg, scenarios),
            total=len(scenarios),
            desc="Concurrent Processing",
        ):
            pass


@timemit
def run_sequential():
    for scenario_name in tqdm(scenarios, desc="Sequential Processing"):
        run_simulation_single_arg(scenario_name)


# Register the cleanup function to be called at exit
def cleanup():
    logfile.close()


atexit.register(cleanup)


def main():
    print(
        f"Starting {simulation_name} simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        file=logfile,
    )

    run_sequential()

    print(
        f"Finished {simulation_name} simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        file=logfile,
    )


if __name__ == "__main__":
    main()
