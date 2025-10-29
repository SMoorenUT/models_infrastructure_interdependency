import pandas as pd
from pathlib import Path

WEIGHT_FACTOR_CARGO = 2

# Initialize the data directories
BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "output_simulations/ema_road_model_17_07_2025/road_network"


def load_dataframes():
    # Load the data
    df_cargo_vkt = pd.read_csv(DATA_DIR / "cargo_vkt.csv", header=0, index_col=0)
    df_passenger_vkt = pd.read_csv(
        DATA_DIR / "passenger_vkt.csv", header=0, index_col=0
    )
    return df_cargo_vkt, df_passenger_vkt


def combine_dataframes(weight_factor_cargo=1):
    df_cargo_vkt, df_passenger_vkt = load_dataframes()
    df_combined_vkt = df_cargo_vkt * weight_factor_cargo + df_passenger_vkt
    print(df_cargo_vkt.tail())
    print(df_passenger_vkt.tail())
    print(df_combined_vkt.tail())
    return df_combined_vkt


def save_dataframe(df):
    df.to_csv(DATA_DIR / "combined_vkt.csv")


def main():
    combined_df = combine_dataframes(WEIGHT_FACTOR_CARGO)
    save_dataframe(combined_df)


if __name__ == "__main__":
    main()
