import run_simulation
import pathlib

scenario_stem = "data/scenarios_ema/ema_road_model_07_05_2024_scenario_"
NUM_SIMULATIONS = 2

for i in range(NUM_SIMULATIONS):
    scenario_config_file = scenario_stem + f"{i}.json"
    output_dir = scenario_stem + f"{i}"
    run_simulation.run_simulation(scenario_config_file, output_dir)