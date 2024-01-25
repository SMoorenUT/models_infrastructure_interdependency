import numpy as np
import pandas as pd
import pathlib
from scipy.interpolate import CubicSpline
from typing import Union, List
from tape_creator_functions import create_lists_sampling_input, latin_hypercube_sampling, cubic_spline_interpolation, find_unique_values, normalize_df


CURR_DIR = pathlib.Path(__file__).parent
POPULATION_CSV = CURR_DIR.joinpath(
    "input_data/03759ned_UntypedDataSet_04122023_152457.csv"
)
# List of population file names
population_file_names = [
    "Bevolking__intervallen__regio__2023_2050_07122023_164139_bovengrens.csv",
    "Bevolking__intervallen__regio__2023_2050_07122023_164153_ondergrens.csv",
    "Bevolking__intervallen__regio__2023_2050_07122023_164205_prognose.csv",
    "Bevolking__intervallen__regio__2023_2050_07122023_170354_bovengrens_2.csv",
    "Bevolking__intervallen__regio__2023_2050_07122023_170348_ondergrens_2.csv",
    "Bevolking__intervallen__regio__2023_2050_07122023_170341_prognose_2.csv",
]

file_names_old = [
    "Bevolking__intervallen__regio__2023_2050_27112023_163836_bovengrens.csv",
    "Bevolking__intervallen__regio__2023_2050_27112023_163822_ondergrens.csv",
    "Bevolking__intervallen__regio__2023_2050_27112023_162259_prognose.csv",
]


municipalities = [
    "Goeree-Overflakkee",
    "Veenendaal",
    "Molenlanden",
    "Gooise Meren",
    "Westland",
    "Midden-Delfland",
    "Zuidplas",
    "Rotterdam",
    "Stichtse Vecht",
    "Nieuwkoop",
    "Alkmaar",
    "Goeree-Overflakkee",
    "Hellevoetsluis",
    "Nissewaard",
    "Gooise Meren",
    "Heemskerk",
    "Blaricum",
    "Edam-Volendam",
    "Bunnik",
    "Bunschoten",
    "Leusden",
    "Woudenberg",
    "Velsen",
    "Bergen (NH.)",
    "Castricum",
    "De Ronde Venen",
    "Woerden",
    "'s-Gravenhage",
    "Katwijk",
    "Ouder-Amstel",
    "Hoeksche Waard",
    "Papendrecht",
    "Teylingen",
    "Lansingerland",
    "Bodegraven-Reeuwijk",
    "Krimpenerwaard",
    "Amersfoort",
    "Goeree-Overflakkee",
    "Westvoorne",
    "Goeree-Overflakkee",
    "Eemnes",
    "Albrandswaard",
    "Hoeksche Waard",
    "Goeree-Overflakkee",
    "Utrecht",
    "Goeree-Overflakkee",
    "Wijdemeren",
    "Hellevoetsluis",
    "Hoeksche Waard",
    "Goeree-Overflakkee",
    "'s-Gravenhage",
    "Zwijndrecht",
    "Goeree-Overflakkee",
    "Blaricum",
    "Gorinchem",
    "Blaricum",
    "Hendrik-Ido-Ambacht",
    "Zeist",
    "Schiedam",
    "Vlaardingen",
    "Hoeksche Waard",
    "Rijswijk",
    "Capelle aan den IJssel",
    "Huizen",
    "Waterland",
    "Utrechtse Heuvelrug",
    "Vijfheerenlanden",
    "Goeree-Overflakkee",
    "Heemstede",
    "Waddinxveen",
    "Amsterdam",
    "Diemen",
    "Lopik",
    "Bloemendaal",
    "Gooise Meren",
    "Gouda",
    "Haarlemmermeer",
    "Hoeksche Waard",
    "Velsen",
    "Goeree-Overflakkee",
    "Hellevoetsluis",
    "Goeree-Overflakkee",
    "Barendrecht",
    "Zandvoort",
    "Goeree-Overflakkee",
    "Zoeterwoude",
    "Lisse",
    "Goeree-Overflakkee",
    "Brielle",
    "Hilversum",
    "Heiloo",
    "Renswoude",
    "Leidschendam-Voorburg",
    "Wormerland",
    "Gooise Meren",
    "Goeree-Overflakkee",
    "Goeree-Overflakkee",
    "Uithoorn",
    "Goeree-Overflakkee",
    "Goeree-Overflakkee",
    "Beemster",
    "Zaanstad",
    "Uitgeest",
    "De Bilt",
    "Hardinxveld-Giessendam",
    "Landsmeer",
    "Goeree-Overflakkee",
    "Goeree-Overflakkee",
    "Sliedrecht",
    "Dordrecht",
    "Amstelveen",
    "Amsterdam",
    "Goeree-Overflakkee",
    "Gooise Meren",
    "Beverwijk",
    "Hellevoetsluis",
    "Langedijk",
    "Haarlem",
    "Weesp",
    "Wassenaar",
    "Amsterdam",
    "Ridderkerk",
    "Krimpen aan den IJssel",
    "Delft",
    "Laren",
    "Noordwijk",
    "Alphen aan den Rijn",
    "Heerhugowaard",
    "Goeree-Overflakkee",
    "Voorschoten",
    "Leiderdorp",
    "Pijnacker-Nootdorp",
    "Hoeksche Waard",
    "Goeree-Overflakkee",
    "Goeree-Overflakkee",
    "Oegstgeest",
    "Gooise Meren",
    "Oudewater",
    "Baarn",
    "Goeree-Overflakkee",
    "Goeree-Overflakkee",
    "Leiden",
    "Hillegom",
    "Purmerend",
    "Amsterdam",
    "Amsterdam",
    "Diemen",
    "Montfoort",
    "Amsterdam",
    "Huizen",
    "Oostzaan",
    "Maassluis",
    "Kaag en Braassem",
    "Rhenen",
    "Soest",
    "Alblasserdam",
    "Nieuwegein",
    "Zoetermeer",
    "Wijk bij Duurstede",
    "Goeree-Overflakkee",
    "IJsselstein",
    "Goeree-Overflakkee",
    "Aalsmeer",
    "Hoeksche Waard",
    "Houten",
]  # List of municipalities as ordered in municipalities_area_set.json
municipalities_unique = sorted(
    pd.Series(municipalities).unique().tolist()
)  # Sorted list with unique municipalities (in simulation)

municipality_rename_dict = {
    "'s-Gravenhage (gemeente)": "'s-Gravenhage",
    "Laren (NH.)": "Laren",
    "Rijswijk (ZH.)": "Rijswijk",
    "Utrecht (gemeente)": "Utrecht",
}

corop_areas_study_area = [
    "Utrecht",
    "Alkmaar en omgeving",
    "IJmond",
    "Agglomeratie Haarlem",
    "Zaanstreek",
    "Groot-Amsterdam",
    "Het Gooi en Vechtstreek",
    "Agglomeratie Leiden en Bollenstreek",
    "Agglomeratie 's-Gravenhage",
    "Delft en Westland",
    "Oost-Zuid-Holland",
    "Groot-Rijnmond",
    "Zuidoost-Zuid-Holland",
]

def get_corop_dictionary():
    # Read the tables from the webpage. The table we're interested in is the first one
    corop_table = pd.read_html("https://nl.wikipedia.org/wiki/COROP")[0]

    # Create a dictionary where each key is a COROP area and each value is a list of municipalities in that area
    corop_dict = corop_table.groupby("COROP-gebied")["Gemeenten"].apply(list).to_dict()

    # Drop all keys that are not in the jobs.df.iloc[:, 0] list
    corop_dict = {key: value for key, value in corop_dict.items() if key in corop_areas_study_area}
    
    # Create a list of all municipalities in the column "Gemeenten" in corop_dict
    corop_municipalities = []
    for value in corop_dict.values():
        corop_municipalities.extend(value)

    return corop_dict, corop_municipalities

# Function to read CSV files
def read_csv(file_name, delimiter=";"):
    try:
        # Use ";" as the delimiter
        df = pd.read_csv(file_name, delimiter=delimiter)
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"File '{file_name}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file '{file_name}'. Check the delimiter.")
        return None


def read_excel(file_name):
    try:
        # Try reading with UTF-8 encoding
        df = pd.read_excel(file_name)
        return df
    except UnicodeDecodeError:
        # If decoding with UTF-8 fails, try other encodings
        try_encodings = ["ISO-8859-1", "cp1252"]
        for encoding in try_encodings:
            try:
                df = pd.read_excel(file_name, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        # If all attempts fail, raise an exception
        raise Exception(
            f"Unable to decode file '{file_name}' with available encodings."
        )

# Read each CSV file future population
dfs_bevolking_prognose = [
    read_csv(CURR_DIR / "input_data" / file_name) for file_name in population_file_names
]

# Filter out None values (in case of errors)
dfs_bevolking_prognose = [df for df in dfs_bevolking_prognose if df is not None]


def concatenate_population_dfs(dfs):
    indices = [0, 1, 2]
    for i, j in zip(indices, range(3, 6)):
        dfs[i] = pd.concat([dfs[i], dfs[j]], ignore_index=True)
    dfs = dfs[:3]
    return dfs


dfs_bevolking_prognose = concatenate_population_dfs(dfs_bevolking_prognose)

# Sort df on municipality name and year, then reset index
dfs_bevolking_prognose = [
    df.sort_values(by=["Regio-indeling 2021", "Perioden"]).reset_index(drop=True)
    for df in dfs_bevolking_prognose
]


# Function to apply renaming to a DataFrame
def rename_municipalities(df):
    df["Regio-indeling 2021"] = df["Regio-indeling 2021"].replace(
        municipality_rename_dict
    )
    return df


# Apply the renaming function to each DataFrame in the list
dfs_bevolking_prognose = [rename_municipalities(df) for df in dfs_bevolking_prognose]

# Filter rows based on the municipalities list
dfs_filtered = [
    df[df["Regio-indeling 2021"].isin(municipalities_unique)]
    for df in dfs_bevolking_prognose
]


# Create a new DataFrame with columns from "...ondergrens" and an additional column from "...bovengrens"
df_final = pd.DataFrame()


# Extract the relevant columns from "ondergrens"
cols_ondergrens = [
    "Prognose(-interval)",
    "Leeftijd",
    "Perioden",
    "Regio-indeling 2021",
    "Totale bevolking (x 1 000)",
]
# Extract the corresponding DataFrame from "bovengrens"
df_ondergrens = dfs_bevolking_prognose[1][cols_ondergrens]
df_bovengrens = dfs_bevolking_prognose[0]

# Create a new column for merging by concatenating "Regio-indeling 2021" and "Perioden"
regio_indeling_perioden_column = (
    df_ondergrens["Regio-indeling 2021"] + "_" + df_ondergrens["Perioden"].astype(str)
)
# Insert the new column at index 1
df_ondergrens.insert(
    loc=1, column="Regio-indeling_Perioden", value=regio_indeling_perioden_column
)
# Create a new column for merging by concatenating "Regio-indeling 2021" and "Perioden"
df_bovengrens["Regio-indeling_Perioden"] = (
    df_bovengrens["Regio-indeling 2021"] + "_" + df_bovengrens["Perioden"].astype(str)
)

# Merge the two DataFrames based on common columns
df_merged = pd.merge(
    df_ondergrens,
    df_bovengrens[["Regio-indeling_Perioden", "Totale bevolking (x 1 000)"]],
    on="Regio-indeling_Perioden",
)

# Rename columns
df_merged = df_merged.rename(
    columns={
        "Totale bevolking (x 1 000)_x": "Ondergrens totale bevolking (x 1 000)",
        "Totale bevolking (x 1 000)_y": "Bovengrens totale bevolking (x 1 000)",
        "Regio-indeling 2021": "Gemeentenaam",
        "Perioden": "Year",
    }
)

# Drop gemeentes that are not in the list of municipalities
df_merged = df_merged[df_merged["Gemeentenaam"].isin(municipalities_unique)]

# Drop unnecessary columns
df_final = df_merged.drop(columns=["Prognose(-interval)", "Leeftijd"])

# Now 'df_final' contains a new DataFrame with columns from "...ondergrens" and an additional column from "...bovengrens"


def prepare_historic_population_df(municipalities_unique):
    # Read csv's for population
    df_bevolking_2019_2023 = read_csv(POPULATION_CSV)

    # Read the Excel file with municipality names
    df_gemeentes = read_excel(
        CURR_DIR.joinpath("input_data/gemeenten-alfabetisch-2021.xlsx")
    )

    # Merge dataframes so that the municipality names are human readable in df_bevolking_2019_2023
    df_bevolking_2019_2023 = pd.merge(
        df_bevolking_2019_2023,
        df_gemeentes[["GemeentecodeGM", "Gemeentenaam"]],
        left_on="RegioS",
        right_on="GemeentecodeGM",
        how="inner",
    ).drop(columns="GemeentecodeGM")

    # Drop rows where "Gemeentenaam" is in the list of municipalities
    df_bevolking_2019_2023 = df_bevolking_2019_2023[
        df_bevolking_2019_2023["Gemeentenaam"].isin(municipalities_unique)
    ]

    # Clean up table
    df_bevolking_2019_2023 = df_bevolking_2019_2023[
        ["ID", "Perioden", "BevolkingOp1Januari_1", "Gemeentenaam"]
    ]  # drops unnecessary columns
    df_bevolking_2019_2023["Year"] = (
        df_bevolking_2019_2023["Perioden"].str[:4].astype(int)
    )
    df_bevolking_2019_2023 = df_bevolking_2019_2023.drop(columns=["Perioden"])
    # Divide population by 1000 to get the same unit as the other dataframes
    df_bevolking_2019_2023["BevolkingOp1Januari_1"] = (
        df_bevolking_2019_2023["BevolkingOp1Januari_1"] / 1000
    )
    # Rename population column
    df_bevolking_2019_2023 = df_bevolking_2019_2023.rename(
        columns={"BevolkingOp1Januari_1": "Population (x 1 000)"}
    )

    return df_bevolking_2019_2023


df_bevolking_2019_2023 = prepare_historic_population_df(municipalities_unique)

# Check if all municipalities are present in both DataFrames
municipalities_df_final = df_final["Gemeentenaam"].unique()
municipalities_df_2019_2030 = df_bevolking_2019_2023["Gemeentenaam"].unique()
find_unique_values(municipalities_df_final, municipalities_df_2019_2030)


def combine_historic_population_with_prognosis(df_prognosis, df_bevolking_2019_2023):
    # List of years to add before each municipality
    years_to_add = [2019, 2020, 2021, 2022, 2023]

    # Create a list of dictionaries for the new rows
    new_rows = []
    municipalities_df_final = df_final["Gemeentenaam"].unique()
    for municipality in municipalities_df_final:
        for year in years_to_add:
            new_row = {
                "Gemeentenaam": municipality,
                "Year": year,
                # Add other columns as needed
            }
            new_rows.append(new_row)

    # Create a new DataFrame from the list of dictionaries
    df_new_rows = pd.DataFrame(new_rows)

    # Concatenate the new DataFrame with the existing DataFrame
    df_bevolking_extended = pd.concat(
        [df_prognosis, df_bevolking_2019_2023], ignore_index=True
    )

    # Sort the DataFrame by "Gemeentenaam" and "Year" columns
    df_bevolking_extended = df_bevolking_extended.sort_values(
        by=["Gemeentenaam", "Year"]
    ).reset_index(drop=True)

    # Fill Regio-indeling_Perioden column
    df_bevolking_extended["Regio-indeling_Perioden"] = (
        df_bevolking_extended["Gemeentenaam"]
        + "_"
        + df_bevolking_extended["Year"].astype(str)
    )

    # Rename Regio-indeling_Perioden to Gemeentenaam
    df_bevolking_extended = df_bevolking_extended.rename(
        columns={"Regio-indeling_Perioden": "Gemeente_jaar"}
    )

    return df_bevolking_extended


df_bevolking_extended = combine_historic_population_with_prognosis(
    df_final, df_bevolking_2019_2023
)


def create_population_dict_sample(df_bevolking):
    # Create a dictionary with the population of each municipality for each year
    population_scenario_dict = {}
    for municipality in municipalities_unique:
        population_scenario_dict[municipality] = {}
        for year in range(2019, 2051):
            population_scenario_dict[municipality][year] = None

    # Fill the dictionary with the historic (2019-2023) population values from the DataFrame
    for index, row in df_bevolking.iterrows():
        municipality = row["Gemeentenaam"]
        year = row["Year"]
        population = row["Population (x 1 000)"]
        population_scenario_dict[municipality][year] = population

    # Sample 2025-2050 population values from the DataFrame every 5 years between ondergrens and bovengrens column
    for index, row in df_bevolking.iterrows():
        municipality = row["Gemeentenaam"]
        year = row["Year"]
        ondergrens = row["Ondergrens totale bevolking (x 1 000)"]
        bovengrens = row["Bovengrens totale bevolking (x 1 000)"]
        # Sample 2025-2050 population values from the DataFrame every 5 years between ondergrens and bovengrens column
        if year in range(2025, 2051, 5):
            population_scenario_dict[municipality][year] = np.random.uniform(
                ondergrens, bovengrens
            )

    # Convert NaN values to None
    population_scenario_dict = {
        municipality: {
            year: (None if pd.isna(population) else population)
            for year, population in years.items()
        }
        for municipality, years in population_scenario_dict.items()
    }

    # Fill the remaining values with cubic spline interpolation
    for municipality in municipalities_unique:
        x = []
        y = []
        for year, population in population_scenario_dict[municipality].items():
            if population is not None:
                x.append(year)
                y.append(population)
        # Create a cubic spline interpolation function
        cs = CubicSpline(x, y)
        # Create a list of years to interpolate
        x_interp = list(range(2019, 2051))
        # Interpolate the population values
        y_interp = cs(x_interp)
        # Replace the population values in the dictionary with the interpolated values
        for year, population in zip(x_interp, y_interp):
            population_scenario_dict[municipality][year] = population

    return population_scenario_dict

def read_jobs_data():
    """
    Read the jobs data from the CSV file and return the DataFrame
    """
    rv = pd.read_csv(CURR_DIR / "input_data" / "banen_inputdata.csv", delimiter=";", index_col=1, header=0, decimal=",")
    rv = rv.drop(columns="aantal banen(x 1000)")
    return rv

def create_dict_for_jobs_sampling(df_jobs, operator="min"):
    """
    Create a dictionary to sample jobs from based on the df_jobs DataFrame
    """
    # Normalize the jobs data
    df_jobs_normalised = normalize_df(df_jobs)

    # Creating the dictionary
    result_dict = {
        "2019": create_lists_sampling_input(df_jobs_normalised, 2019, "min"),
        "2030_min": create_lists_sampling_input(df_jobs_normalised, 2030, "min"),
        "2030_max": create_lists_sampling_input(df_jobs_normalised, 2030, "max"),
        "2050_min": create_lists_sampling_input(df_jobs_normalised, 2050, "min"),
        "2050_max": create_lists_sampling_input(df_jobs_normalised, 2050, "max"),
    }
    return result_dict

def sample_jobs(df_jobs, num_samples=50):
    jobs_sampling_dict = create_dict_for_jobs_sampling(df_jobs, operator="min")
    sampled_jobs_dict = latin_hypercube_sampling(jobs_sampling_dict , num_samples=num_samples)
    return sampled_jobs_dict

# def cubic_spline_interpolation(samples_dict):
#     # Create Data Points
#     x = [2019, 2030, 2050]

#     first_values_dict = {}

#     for scen, years_dict in samples_dict.items():
#         for year, values_list in years_dict.items():
#             for position, key in enumerate(corop_areas_study_area, start=1):
#                 if scen not in first_values_dict:
#                     first_values_dict[scen] = {}

#                 if key not in first_values_dict[scen]:
#                     first_values_dict[scen][key] = []

#                 first_values_dict[scen][key].append(values_list[position - 1])

#     # Make list of scenario's to loop through (['Scenario_0001', 'Scenario_0002', ... , 'Scenario_XXXX'])
#     scenarios = list(first_values_dict.keys())

#     # List of column indices
#     column_indices = np.arange(len(corop_areas_study_area))

#     # Dictionary to store cubic spline objects
#     cs_dict = {}

#     # Loop through scenarios
#     for scenario in scenarios:
#         # Create a nested dictionary for each scenario
#         scenario_dict = {}

#         # Loop through corop areas
#         for column_index in column_indices:
#             y = first_values_dict[scenario][corop_areas_study_area[column_index]]

#             # Perform Cubic Spline Interpolation
#             cs = CubicSpline(x, y)

#             # Evaluate Interpolation Function
#             x_interp = list(range(2019, 2051))
#             y_interp = cs(x_interp) # y_interp is a list of interpolated values as numpy.float64

#             corop_name = corop_areas_study_area[
#                 column_index
#             ]  # Create a name for the corop area (nested inside the scenario)
#             scenario_dict[corop_name] = y_interp

#         # Assign the nested dictionary to the scenario key
#         cs_dict[scenario] = scenario_dict

#     return cs_dict

def create_jobs_scenarions_dict(df_jobs, num_samples=50):
    """
    Create a dictionary with the sampled jobs for each municipality for the years 2019, 2030 and 2050. 
    The years 2030 and 2050 are sampled from the minimum and maximum values of the jobs in the corop areas. 
    The remaining years are interpolated using cubic spline interpolation.
    """
    jobs_dict_sample = sample_jobs(df_jobs, num_samples=num_samples)
    jobs_interpolated_dictionary = cubic_spline_interpolation(jobs_dict_sample, corop_areas_study_area)
    return jobs_interpolated_dictionary

def generate_jobs_data(num_samples):
    df_jobs = read_jobs_data()
    jobs_interpolated_dictionary = create_jobs_scenarions_dict(df_jobs, num_samples)
    return jobs_interpolated_dictionary

def main():
    num_samples = 10
    population = {
        f"Scenario_{i:04d}": create_population_dict_sample(df_bevolking_extended)
        for i in range(num_samples)
    }
    jobs = generate_jobs_data(num_samples)
    corop_dict, corop_municipalities = get_corop_dictionary()
    find_unique_values(municipalities_unique, corop_municipalities)
    return population, jobs


if __name__ == "__main__":
    population, jobs = main()
