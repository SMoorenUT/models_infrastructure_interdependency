import run_simulation
import pathlib
import sys
import atexit
import time

scenario_stem = "data/scenarios/ema_road_model_17_07_2025_experiment_"
NUM_SIMULATIONS = 1000
LEN_SIM = 3
CURRENT_DATE = time.strftime("%Y-%m-%d")

# Extract the last part of the string before '_scenario_'
simulation_name = scenario_stem.split("_scenario_", 1)[0]


def find_highest_numbered_folder(scenario_stem):
    base_dir = pathlib.Path(scenario_stem).parent
    folder_prefix = pathlib.Path(scenario_stem).name
    max_num = None
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(folder_prefix):
            suffix = folder.name[len(folder_prefix) :]
            if suffix.isdigit():
                num = int(suffix)
                if max_num is None or num > max_num:
                    max_num = num
    return max_num


STARTING_NUMBER = find_highest_numbered_folder(scenario_stem)


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
    print(f"Starting simulation experiment {i:0{LEN_SIM}d}...")
    run_simulation.run_simulation(scenario_config_file, output_dir)
