import run_simulation
import multiprocessing
import time

SCENARIO_STEM = "data/scenarios_ema/ema_road_model_08_05_2024_scenario_"
NUM_SIMULATIONS = 1000
LEN_SIM = 3 # Number of digits in the simulation number


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
    if NUM_SIMULATIONS % 5 != 0:
        raise ValueError("NUM_SIMULATIONS must be divisible by 5")

    num_processes_iterations = NUM_SIMULATIONS // 5
    multiples_of_5 = [5 * i for i in range(num_processes_iterations)]

    for processes_iter, processes_value in enumerate(multiples_of_5):
        processes = []
        for i in range(processes_value, processes_value+5):
            process = multiprocessing.Process(target=run_simulation_parallel, args=(i,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

if __name__ == "__main__":
    main()

pass