from pathlib import Path
import pandas as pd
import numpy as np


CURR_DIR = Path(__file__).parent


def load_data():
    data = pd.read_csv(CURR_DIR / "ema_road_model_27_05_2024_results.csv", index_col=0)
    simulation_input_df = data.iloc[:, :-3]
    simulation_output_dict = data.iloc[:, -3:].to_dict()
    simulation_output_dict = {
        key: np.array(list(value.values()))
        for key, value in simulation_output_dict.items()
    }  # Convert the dictionary values to numpy arrays
    return simulation_input_df, simulation_output_dict


def main():
    simulation_input, simulation_output = load_data()



if __name__ == "__main__":
    main()
