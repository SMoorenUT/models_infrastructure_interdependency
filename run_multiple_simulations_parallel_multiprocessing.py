import run_simulation
import multiprocessing
import time

SCENARIO_STEM = "data/scenarios_ema/ema_road_model_27_05_2024_scenario_"
NUM_SIMULATIONS = 100
LEN_SIM = 2
NUMBER_OF_SIMULATIONS_PARALLEL = 5


def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    
def run_simulation_parallel(i):
    scenario_config_file = SCENARIO_STEM + f"{i:0{LEN_SIM}d}.json"
    output_dir = SCENARIO_STEM + f"{i:0{LEN_SIM}d}"
    run_simulation.run_simulation(scenario_config_file, output_dir)
    # print(scenario_config_file)
    # time.sleep(5)
    # print(output_dir)

def run_simulation_parallel_test(i):
    scenario_config_file = SCENARIO_STEM + f"{i:0{LEN_SIM}d}.json"
    output_dir = SCENARIO_STEM + f"{i:0{LEN_SIM}d}"
    print(scenario_config_file)
    time.sleep(1)
    print(output_dir)
    time.sleep(1)

def run_fibonacci_parallel_test(i):
    result = fibonacci(i)
    print(f"Fibonacci({i}) = {result}")

def main():
    if NUM_SIMULATIONS % NUMBER_OF_SIMULATIONS_PARALLEL != 0:
        raise ValueError("NUM_SIMULATIONS must be divisible by {NUMBER_OF_SIMULATIONS_PARALLEL}")

    num_processes_iterations = NUM_SIMULATIONS // NUMBER_OF_SIMULATIONS_PARALLEL
    multiples_number_of_sims_parallel = [NUMBER_OF_SIMULATIONS_PARALLEL * i for i in range(num_processes_iterations)]

    fibonacci_ns = [35, 10, 25, 40, 16, 9, 11, 31, 24, 7]

    for processes_iter, processes_value in enumerate(fibonacci_ns):
        processes = []
        for i in range(processes_value, processes_value + 5):
            process = multiprocessing.Process(target=run_fibonacci_parallel_test, args=(i,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

if __name__ == "__main__":
    main()

pass