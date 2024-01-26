from typing import List, Union
import numpy as np
import pandas as pd
from scipy.stats import uniform
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler

def create_lists_sampling_input(df_for_sampling_input: pd.DataFrame, year: int, operator: str = "min") -> List[Union[int, float]]:
    # Create lists from sampling df
    year_cols = {2019: [0], 2030: [1, 2], 2050: [3, 4]}

    if year not in year_cols:
        raise ValueError("Invalid year")

    cols = year_cols[year]

    if operator not in {"min", "max"}:
        raise ValueError("Invalid operator. Must be 'min' or 'max'")

    return getattr(df_for_sampling_input.iloc[:, cols], operator)(axis=1).tolist()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    last_columns = df.columns[-4:]
    first_column = df.columns[0]
    df[last_columns] = df[last_columns].div(df[first_column], axis=0)
    df[first_column] = 1
    return df

def establish_length_num_samples(num_samples: int):
    if num_samples <= 0:
        ValueError("num_samples must be greater than 0")
    length_num_samples = len(str(num_samples)) if num_samples not in [10**i for i in range(10)] else len(str(num_samples)) - 1 # To format het scenario names with leading zeros
    return length_num_samples

def latin_hypercube_sampling(input_sample_dict: dict, num_samples: int = 100) -> dict:
    # Set the random seed for reproducibility
    np.random.seed(0)

    # Define lower and upper bounds
    lower_bound_2030 = input_sample_dict["2030_min"]  # Lower bound for each dimension in 2030
    upper_bound_2030 = input_sample_dict["2030_max"]  # Upper bound for each dimension in 2030
    lower_bound_2050 = input_sample_dict["2050_min"]  # Lower bound for each dimension in 2050
    upper_bound_2050 = input_sample_dict["2050_max"]  # Upper bound for each dimension in 2050

    # Define the number of samples and dimensions
    num_samples = num_samples
    num_dimensions = (
        len(lower_bound_2030)
        if len(lower_bound_2030)
        == len(upper_bound_2030)
        == len(lower_bound_2050)
        == len(upper_bound_2050)
        else None
    )
    length_num_samples = establish_length_num_samples(num_samples)

    # Generate Latin Hypercube Samples in dictionary using scipy.stats.uniform
    lhs_samples_dict = {
        f"Scenario_{i:0{length_num_samples}d}": {
            "2019": input_sample_dict["2019"],
            "2030": [
                uniform(
                    loc=lower_bound_2030[j],
                    scale=upper_bound_2030[j] - lower_bound_2030[j],
                ).rvs()
                for j in range(num_dimensions)
            ],
            "2050": [
                uniform(
                    loc=lower_bound_2050[j],
                    scale=upper_bound_2050[j] - lower_bound_2050[j],
                ).rvs()
                for j in range(num_dimensions)
            ],
        }
        for i in range(num_samples)
    }

    return lhs_samples_dict


def cubic_spline_interpolation(samples_dict, columns):
    # Create Data Points
    x = [2019, 2030, 2050]

    ## Transform the dictionary to {
    #   Scenario1:
    #       {Variable1: [2019value, 2030value, 2050value],
    #       Variable2: [2019value, 2030value, 2050value],
    #       VariableN:[..]
    #       },
    #   ScenarioN:
    #       {VariableN:...}
    #   }

    ##  Or transform the dictionary to {
    #   Scenario1:
    #       {corop1: [2019value, 2030value, 2050value],
    #       corop2: [2019value, 2030value, 2050value],
    #       coropN:[..]
    #       },
    #   ScenarioN:
    #       {coropN:...}
    #   }
    first_values_dict = {}

    for scen, years_dict in samples_dict.items():
        for year, values_list in years_dict.items():
            for position, key in enumerate(columns, start=1):
                if scen not in first_values_dict:
                    first_values_dict[scen] = {}

                if key not in first_values_dict[scen]:
                    first_values_dict[scen][key] = []

                first_values_dict[scen][key].append(values_list[position - 1])

    # Make list of scenario's to loop through (['Scenario_0001', 'Scenario_0002', ... , 'Scenario_XXXX'])
    scenarios = list(first_values_dict.keys())

    # List of column indices
    column_indices = np.arange(len(columns))

    # Dictionary to store cubic spline objects
    cs_dict = {}

    # Loop through scenarios
    for scenario in scenarios:
        # Create a nested dictionary for each scenario
        scenario_dict = {}

        # Loop through columns
        for column_index in column_indices:
            y = first_values_dict[scenario][columns[column_index]]

            # Perform Cubic Spline Interpolation
            cs = CubicSpline(x, y)

            # Evaluate Interpolation Function
            x_interp = list(range(2019, 2051))
            y_interp = cs(x_interp)

            column_name = columns[
                column_index
            ]  # Create a name for the variable (nested inside the scenario)
            scenario_dict[column_name] = y_interp

        # Assign the nested dictionary to the scenario key
        cs_dict[scenario] = scenario_dict

    return cs_dict

def find_unique_values(list1, list2):
    unique_in_list1 = set(list1) - set(list2)
    unique_in_list2 = set(list2) - set(list1)

    result = {
        "Unique in List1": list(unique_in_list1),
        "Unique in List2": list(unique_in_list2),
    }
    
    if unique_in_list1 or unique_in_list2:
        print(result)
    
    return result