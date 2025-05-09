import concurrent.futures
import json
import math
from pathlib import Path
import tempfile
import os
from movici_simulation_core import Simulation
import concurrent.futures
import math
import time
from run_simulation import run_simulation


PRIMES = list(range(1, 100000))

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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            pass


@timemit
def run_sequential():
    for number in PRIMES:
        prime = is_prime(number)
        pass

TASKS = [
    "./data/experiments/experiment_001.json",
    "./data/experiments/experiment_002.json",
]


def run_simulation_single_arg(experiment_file):
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


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for task, _ in zip(TASKS, executor.map(run_simulation, TASKS)):
            print(f"Completed simulation for {task}")


if __name__ == "__main__":
    run_concurrent()
    run_sequential()
