import pandas as pd
from pathlib import Path

WEIGHT_FACTOR_CARGO = 2

# Initialize the data directories
BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "output_simulations/ema_road_model_17_07_2025/road_network"
CARGO_DF_FILENAME = "cargo_demand.csv"
PASSENGER_DF_FILENAME = "passenger_demand.csv"
OUTPUT_FILENAME = "combined_demand.csv"


def load_dataframes():
    # Load the data
    df_cargo_vkt = pd.read_csv(DATA_DIR / CARGO_DF_FILENAME, header=0, index_col=0)
    df_passenger_vkt = pd.read_csv(
        DATA_DIR / PASSENGER_DF_FILENAME, header=0, index_col=0
    )
    return df_cargo_vkt, df_passenger_vkt


def combine_dataframes(weight_factor_cargo=1):
    df_cargo_vkt, df_passenger_vkt = load_dataframes()
    df_combined_vkt = df_cargo_vkt * weight_factor_cargo + df_passenger_vkt
    print(df_cargo_vkt.tail())
    print(df_passenger_vkt.tail())
    print(df_combined_vkt.tail())
    return df_combined_vkt


def save_dataframe(df, output_filename):
    df.to_csv(DATA_DIR / output_filename)


def main():
    combined_df = combine_dataframes(WEIGHT_FACTOR_CARGO)
    save_dataframe(combined_df, output_filename=OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
