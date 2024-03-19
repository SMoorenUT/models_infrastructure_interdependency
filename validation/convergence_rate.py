import numpy as np
import pathlib
import os
import datetime as dt


CURR_DIR = pathlib.Path(__file__).parent
ROOT_DIR = CURR_DIR.parent
DISAGGREGATED_OUTPUT_FOLDER = (
    f"{ROOT_DIR}/data/scenarios/test_ema_20240306_output_disaggregated"
)
MAX_ITERATIONS = 51


def convergence_check(path_to_disaggregated_output_folder: pathlib.Path):
    """
    Calculate the convergence rate of a simulation for a single year
    """
    path = path_to_disaggregated_output_folder
    # Create a list of all filesnames in the folder
    filenames = [filename for filename in os.listdir(path)]
    # Sort list of filenames based on the timestamp first and then interation number
    filenames.sort(key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1])))

    # Create a list of all unique timestamps
    timestamps = list(set([int(filename.split("_")[0][1:]) for filename in filenames]))
    timestamps.sort()

    ## Print all timestamps in human readable format
    # print_timestamps(timestamps)


    # Initialize variables
    number_of_times_cut_off_at_max_iterations = 0

    # Loop through the list of timestamps and calculate the convergence rate for each timestamp
    for i, timestamp in enumerate(timestamps):
        # Create a list of all filenames for the current timestamp
        filenames = [
            filename
            for filename in os.listdir(path)
            if filename.startswith(f"t{timestamp}_")
        ]
        # Sort list of filenames based on the iteration number
        filenames.sort(key=lambda x: int(x.split("_")[1]))
        # Create a list of all unique iteration numbers
        iterations = len(filenames)
        
        if iterations == MAX_ITERATIONS:
            max_iterations_reached = True
            number_of_times_cut_off_at_max_iterations += 1
        else:
            max_iterations_reached = False
        
        print(
            f"Timestamp: t{timestamp}, Iterations: {iterations}, Max iterations reached: {max_iterations_reached}"
        )
    print(f"Total number of timestamps: {len(timestamps)}")
    print(f"Number of times cut off at max iterations: {number_of_times_cut_off_at_max_iterations}")


def print_timestamps(timestamp_list: list[int]):
    """
    Print all timestamps in human readable format
    """
    print("Timestamps:")
    for i, timestamp in enumerate(timestamp_list):
        timestamp = timestamp + 1546333200
        print(f"{i+1}. {timestamp}: {dt.datetime.fromtimestamp(timestamp)}")


if __name__ == "__main__":
    convergence_check(DISAGGREGATED_OUTPUT_FOLDER)
