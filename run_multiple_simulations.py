import run_simulation
import pathlib

scenario_config_file = pathlib.Path("data/scenarios_ema/ema_road_model_19_03_2024_scenario_0.json")
output_dir = scenario_config_file.remove_suffix(".json")
NUM_SIMULATIONS = 10

for i in range(NUM_SIMULATIONS):
    run_simulation.run_simulation(scenario_config_file, output_dir)
    output_dir = scenario_config_file.remove_suffix(".json").joinpath(f"_{i}")
run_simulation.run_simulation(scenario_config_file, output_dir)
