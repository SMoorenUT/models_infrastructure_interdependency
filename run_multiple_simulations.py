import run_simulation
import pathlib

scenario_stem = "data/scenarios_ema/ema_road_model_27_05_2024_scenario_"
NUM_SIMULATIONS = 25
LEN_SIM = 2

for i in range(NUM_SIMULATIONS):
    i = i+76
    scenario_config_file = scenario_stem + f"{i:0{LEN_SIM}d}.json"
    output_dir = scenario_stem + f"{i:0{LEN_SIM}d}"
    run_simulation.run_simulation(scenario_config_file, output_dir)