import pandas as pd
from pathlib import Path

WEIGHT_FACTOR_CARGO = 2

# Initialize the data directories
BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "output_simulations/ema_road_model_27_05_2024/road_network"


def load_dataframes():
    # Load the data
    df_cargo_vkm = pd.read_csv(DATA_DIR / "cargo_vkm.csv", header=0, index_col=0)
    df_passenger_vkm = pd.read_csv(
        DATA_DIR / "passenger_vkm.csv", header=0, index_col=0
    )
    return df_cargo_vkm, df_passenger_vkm


def combine_dataframes(weight_factor_cargo=1):
    df_cargo_vkm, df_passenger_vkm = load_dataframes()
    df_combined_vkm = df_cargo_vkm * weight_factor_cargo + df_passenger_vkm
    print(df_cargo_vkm.tail())
    print(df_passenger_vkm.tail())
    print(df_combined_vkm.tail())
    return df_combined_vkm


def save_dataframe(df):
    df.to_csv(DATA_DIR / "combined_vkm.csv")


def main():
    combined_df = combine_dataframes(WEIGHT_FACTOR_CARGO)
    save_dataframe(combined_df)


if __name__ == "__main__":
    main()
