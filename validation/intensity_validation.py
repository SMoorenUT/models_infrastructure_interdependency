from pathlib import Path
import pandas as pd

# intensity_validation.py


BASE_DIR = Path(__file__).parent
DEFAULT_FILENAME = "intensiteit_blankenburgverbinding_2025.csv"
SIM_NAME = "ema_road_model_17_07_2025"
ROAD_SEGMENT = "3228"

counting_locations = {
    "RWS01_MONIBAS_0040vwm0728ra": "A4Za",
    "RWS01_MONIBAS_0040vwn0728ra": "A4Na",
    "RWS01_MONIBAS_0041hrl0728ra": "A4Nb",
    "RWS01_MONIBAS_0041hrr0728ra": "A4Zb",
    "RWS01_MONIBAS_0151hrl0406ra_1": "A15N",
    "RWS01_MONIBAS_0151hrl0409ra": "A15N2",
    "RWS01_MONIBAS_0151hrr0409ra": "A15Z", 
    "RWS01_MONIBAS_0151hrl0414ra": "A15N3",
    "RWS01_MONIBAS_0151hrr0414ra": "A15Z2",
    "RWS01_MONIBAS_0201hrl0202ra": "A20W",
    "RWS01_MONIBAS_0201hrr0202ra": "A20O",
    "RWS01_MONIBAS_0241hrr0040ra": "A24ZW",
    "RWS01_MONIBAS_0241hrl0040ra": "A24NO",
    "RWS01_MONIBAS_0151hrr0406ra_1": "A15Z"
}

def load_intensity_df(path: Path | str | None = None, filename: str | None = None) -> pd.DataFrame:
    """
    Load the intensity CSV into a pandas DataFrame.
    Tries common European CSV variants if the first attempt fails.
    """
    csv_path = Path(path) if path else BASE_DIR / "data" / (filename if filename else DEFAULT_FILENAME)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Primary attempt: standard comma, dot decimal
    try:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8", low_memory=False)
        return df
    except Exception:
        pass

    # Secondary attempt: semicolon separator, comma decimal (common in Dutch CSVs)
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", decimal=",", low_memory=False)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {exc}") from exc

def combine_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine lane columns into total intensity columns.
    The column "rijstrook_rijbaan" indicates the lane number.
    For each unique id_meetlocatie, start_meetperiode, and eind_meetperiode, sum the intensities across lanes.

    1. Identify unique combinations of id_meetlocatie, start_meetperiode, and eind_meetperiode.
    2. For each combination, sum the intensities across all lanes.
    3. Create new columns for total intensities and drop lane-specific columns.
    4. Return the modified DataFrame.
    """
    # intensity column name
    intensity_col = "gem_intensiteit"

    id_cols = ["id_meetlocatie", "start_meetperiode", "eind_meetperiode"]
    # Group and sum the single intensity column directly (avoids creating a list/extra dataframe)
    grouped_sum = (
        df.groupby(id_cols, as_index=False)[intensity_col]
          .sum()
          .rename(columns={intensity_col: f"{intensity_col}_totaal"})
    )
    return grouped_sum

def extract_date_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date and time from start_meetperiode and eind_meetperiode columns.
    Assumes the format is 'YYYY-MM-DD HH:MM:SS'.
    """
    df["start_date"] = pd.to_datetime(df["start_meetperiode"]).dt.date
    df["start_time"] = pd.to_datetime(df["start_meetperiode"]).dt.time
    df["end_date"] = pd.to_datetime(df["eind_meetperiode"]).dt.date
    df["end_time"] = pd.to_datetime(df["eind_meetperiode"]).dt.time
    return df

def print_df_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the DataFrame.
    """
    print("DataFrame Info:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())

def aggregate_intensity_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total intensity by date.
    """
    aggregated_df = df.groupby(["start_date", "id_meetlocatie"], as_index=False)["gem_intensiteit_totaal"].mean()
    return aggregated_df

def aggregate_intensity_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total intensity by month.
    """
    df["day"] = pd.to_datetime(df["start_date"]).dt.strftime("%d")
    df["month"] = pd.to_datetime(df["start_date"]).dt.strftime("%m")
    df["year"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y")
    aggregated_df = df.groupby(["month", "id_meetlocatie"], as_index=False)["gem_intensiteit_totaal"].mean().round(2)
    return aggregated_df

def load_and_process_intensity_counts(path: Path | str | None = None, filename: str | None = None) -> pd.DataFrame:
    measured_intensity_df = load_intensity_df(filename=filename)
    print("Original DataFrame Info:")
    print_df_info(measured_intensity_df)
    
    # Process data
    measured_intensity_df = combine_lanes(measured_intensity_df)
    measured_intensity_df = extract_date_and_time(measured_intensity_df)
    measured_intensity_df = aggregate_intensity_by_date(measured_intensity_df)
    measured_intensity_df = aggregate_intensity_by_month(measured_intensity_df)
    measured_intensity_df["location_code"] = measured_intensity_df["id_meetlocatie"].map(counting_locations)
    # measured_intensity_df = measured_intensity_df.groupby(
    #     ["month", "location_code"], 
    #     as_index=False
    # )["gem_intensiteit_totaal"].sum()

    # Map combined locations
    location_mapping = {
        "A4Za": "A4Z_combined",
        "A4Zb": "A4Z_combined",
        "A4Na": "A4N_combined",
        "A4Nb": "A4N_combined"
    }
    measured_intensity_df["location_code"] = measured_intensity_df["location_code"].map(
        lambda x: location_mapping.get(x, x)
    )

    measured_intensity_df = measured_intensity_df.groupby(
        ["month", "location_code"],
        as_index=False
    )["gem_intensiteit_totaal"].sum()
    print("\nCombined DataFrame Info:")
    print_df_info(measured_intensity_df)
    return measured_intensity_df

def process_measured_data():
    df_2024 = load_and_process_intensity_counts(filename="intensiteit_blankenburgverbinding_2024.csv")
    df_2025 = load_and_process_intensity_counts(filename="intensiteit_blankenburgverbinding_2025.csv")
    
    # Merge both dataframes on date and location
    merged_df = pd.merge(df_2024, df_2025, on=["month", "location_code"], how="outer", suffixes=("_2024", "_2025"))


    # Calculate yearly averages excluding december
    yearly_avg = merged_df[merged_df["month"] != "12"].groupby("location_code", as_index=False).agg({
        "gem_intensiteit_totaal_2024": "mean",
        "gem_intensiteit_totaal_2025": "mean",
    }).round(2)
    yearly_avg["month"] = "yearly_avg_excl_dec"

    # Append yearly averages to the original data
    merged_df = pd.concat([merged_df, yearly_avg], ignore_index=True)

    merged_df["relative_difference"] = (
        merged_df["gem_intensiteit_totaal_2025"]  /
        merged_df["gem_intensiteit_totaal_2024"])
      # percentage difference

    overview_df = merged_df.pivot_table(
        index="month",
        columns="location_code",
        values="relative_difference",
        aggfunc="first"
    ).round(2)
    print("\nRelative Difference Overview (2025 vs 2024):")
    print(overview_df)

    merged_df = merged_df.round(2)
    return merged_df
    
def process_simulated_data():
    simulated_df = pd.read_csv(BASE_DIR.parent / "output_simulations" / SIM_NAME / "road_network" / "road_segments" / f"{SIM_NAME}_RoadSegment_{ROAD_SEGMENT}_transport.passenger_car_unit.csv", index_col=0)

    # Limit to first 500 rows that include the opening of the Blankenburgverbinding
    simulated_df = simulated_df.head(500)

    # Calculate per row 2025 vs 2024 relative difference
    relative_difference_2025_2024 = (
        simulated_df["2024"] /
        simulated_df["2023"]
    ).round(2)
    print(f"Road segment {ROAD_SEGMENT}: {relative_difference_2025_2024.mean().round(2)}") 

    return simulated_df

def main():
    measured_data = process_measured_data()
    simulated_data = process_simulated_data()


if __name__ == "__main__":
    main()