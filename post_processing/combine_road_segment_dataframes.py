import pandas as pd
from pathlib import Path

# Input parameters
SIM_NAME = "ema_road_model_17_07_2025"
ENTITY_NUMBERS = [3260,651,660] # Must be a list of integers representing road segment IDs with a length of at least 2
ATTRIBUTE_NAME = "transport.passenger_car_unit"
WEIGHT_FACTOR_CARGO = 2
# Initialize the data directories
BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / f"output_simulations/{SIM_NAME}/road_network/road_segments"

highway = "A20"
direction = "northbound"
OUTPUT_FILENAME = f"{SIM_NAME}_RoadSegment_{highway}{direction}_{ATTRIBUTE_NAME}.csv"

# Validate ENTITY_NUMBERS
if not isinstance(ENTITY_NUMBERS, list) or len(ENTITY_NUMBERS) < 2 or not all(isinstance(num, int) for num in ENTITY_NUMBERS):
    raise ValueError("ENTITY_NUMBERS must be a list of at least two integers representing road segment IDs.")

def get_filename(entity_number, attribute_name):
    return f"{SIM_NAME}_RoadSegment_{entity_number}_{attribute_name}.csv"

def get_list_of_filenames(entity_numbers, attribute_name):
    return [get_filename(entity_number, attribute_name) for entity_number in entity_numbers]

def load_dataframes(list_of_filenames):
    # Load the data
    filenames = list_of_filenames

    if len(filenames) < 2:
        raise ValueError("Need at least two filenames to build dataframes list.")

    # Load all files into a list of DataFrames
    list_of_dataframes = [pd.read_csv(DATA_DIR / fname, header=0, index_col=0) for fname in filenames]

    return list_of_dataframes


def combine_dataframes(list_of_dataframes):
    # Accept either a list of DataFrames or a list of filenames (strings/Paths)
    if isinstance(list_of_dataframes, (str, Path)) or (isinstance(list_of_dataframes, list) and all(isinstance(x, (str, Path)) for x in list_of_dataframes)):
        # treat as filenames -> load into DataFrames
        list_of_dataframes = load_dataframes(list_of_dataframes)
    if not isinstance(list_of_dataframes, list) or len(list_of_dataframes) < 2:
        raise ValueError("combine_dataframes expects a list of at least two DataFrames or filenames.")

    # Sum all DataFrames element-wise, tolerating mismatched indexes/columns
    df_sum = list_of_dataframes[0].copy()
    for df in list_of_dataframes[1:]:
        df_sum = df_sum.add(df, fill_value=0)

    print(df_sum.tail())
    return df_sum



def save_dataframe(df, output_filename):
    df.to_csv(DATA_DIR / output_filename)


def main():
    list_of_filenames = get_list_of_filenames(ENTITY_NUMBERS, ATTRIBUTE_NAME)
    combined_df = combine_dataframes(list_of_filenames)
    save_dataframe(combined_df, output_filename=OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
