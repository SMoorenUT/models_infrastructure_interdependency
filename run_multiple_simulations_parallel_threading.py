import run_simulation
import threading
import run_simulation
import time

SCENARIO_STEM = "data/scenarios_ema/ema_road_model_27_05_2024_scenario_"
NUM_SIMULATIONS = 100
LEN_SIM = 2

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


def main():
    if NUM_SIMULATIONS % 5 != 0:
        raise ValueError("NUM_SIMULATIONS must be divisible by 5")

    num_threads_iterations = NUM_SIMULATIONS // 5
    multiples_of_5 = [5 * i for i in range(num_threads_iterations)]

    for threads_iter, threads_value in enumerate(multiples_of_5):
        threads = []
        for i in range(threads_value, threads_value + 5):
            thread = threading.Thread(target=run_simulation_parallel_test, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()
