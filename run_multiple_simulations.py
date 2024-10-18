import run_simulation
import pathlib
import sys
import atexit
import time

scenario_stem = "data/scenarios_ema_1000/ema_road_model_08_05_2024_scenario_"
NUM_SIMULATIONS = 1000
LEN_SIM = 3
STARTING_NUMBER = 992
CURRENT_DATE = time.strftime("%Y-%m-%d")

# Extract the last part of the string before '_scenario_'
simulation_name = scenario_stem.split("_scenario_", 1)[0]


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

sys.stdout = DualOutput(f"{simulation_name}_started_at_{CURRENT_DATE}.txt")

for i in range(NUM_SIMULATIONS):
    i = i + STARTING_NUMBER
    scenario_config_file = scenario_stem + f"{i:0{LEN_SIM}d}.json"
    output_dir = scenario_stem + f"{i:0{LEN_SIM}d}"
    print(f"Starting simulation scenario {i:0{LEN_SIM}d}...")
    run_simulation.run_simulation(scenario_config_file, output_dir)
