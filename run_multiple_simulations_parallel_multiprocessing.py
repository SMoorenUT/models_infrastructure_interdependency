import run_simulation
import multiprocessing

NUMBER_OF_SIMULATIONS_PARALLEL = 5
SCENARIO_STEM = "data/scenarios_ema_1000/ema_road_model_08_05_2024_scenario_"
NUM_SIMULATIONS = 1000
LEN_SIM = 3 # Number of digits in the simulation number
STARTING_NUMBER = 0
    
def run_simulation_parallel(i):
    scenario_config_file = SCENARIO_STEM + f"{i:0{LEN_SIM}d}.json"
    output_dir = SCENARIO_STEM + f"{i:0{LEN_SIM}d}"
    run_simulation.run_simulation(scenario_config_file, output_dir)


def main():
    if NUM_SIMULATIONS % NUMBER_OF_SIMULATIONS_PARALLEL != 0:
        raise ValueError("NUM_SIMULATIONS must be divisible by {NUMBER_OF_SIMULATIONS_PARALLEL}")

    num_processes_iterations = NUM_SIMULATIONS // NUMBER_OF_SIMULATIONS_PARALLEL
    multiples_number_of_sims_parallel = [NUMBER_OF_SIMULATIONS_PARALLEL * i for i in range(num_processes_iterations)]

    for processes_iter, processes_value in enumerate(multiples_number_of_sims_parallel):
        processes = []
        for i in range(processes_value, processes_value+5):
            i = i + STARTING_NUMBER
            process = multiprocessing.Process(target=run_simulation_parallel, args=(i,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

if __name__ == "__main__":
    main()