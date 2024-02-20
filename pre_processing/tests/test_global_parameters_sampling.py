import numpy as np
import pytest
from pre_processing.global_parameters_sampling import (
    create_global_parameters_scenarios,
    read_csv_file,
    clear_df,
    create_sampling_dictionary,
    add_global_variable,
    add_global_variables,
    populate_dataframe_interpolated,
    save_global_parameters_scenarios_as_csv,
)
import pandas as pd

@pytest.fixture
def num_samples():
    return 10

@pytest.fixture
def output_path(tmp_path):
    return tmp_path

@pytest.fixture
def df_sampling():
    return None

def test_create_global_parameters_scenarios(num_samples, output_path):
    create_global_parameters_scenarios(num_samples=num_samples, output_path=output_path)
    
    # Add assertions to check if the expected output files are created in the output path
    assert (output_path / "scenario_1.csv").exists()
    assert (output_path / "scenario_2.csv").exists()
    # Add more assertions for other scenarios
    
    # Add assertions to check the content of the output files if necessary
    # For example, you can read the CSV files and compare the data with expected values
    
    # assert ...

def test_read_csv_file():
    # Test reading a CSV file
    file_path = "/home/moorens/code/data/init_data/safety_interpolated.csv"
    df = read_csv_file(file_path)
    
    # Add assertions to check if the DataFrame is read correctly
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # assert ...

def test_clear_df():
    # Test clearing a DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    cleared_df = clear_df(df)
    
    # Add assertions to check if the DataFrame is cleared correctly
    assert cleared_df.equals(pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, np.nan], "C": [7, np.nan, np.nan]}))

def test_create_sampling_dictionary():
    # Test creating a sampling dictionary
    df = pd.DataFrame({
        "2019": [1, 2, 3, 4, 5],
        "2030_min": [6, 7, 8, 9, 10],
        "2030_max": [11, 12, 13, 14, 15],
        "2050_min": [16, 17, 18, 19, 20],
        "2050_max": [21, 22, 23, 24, 25]
    })
    sampling_dictionary = create_sampling_dictionary(df)
    
    # Add assertions to check if the sampling dictionary is created correctly
    assert isinstance(sampling_dictionary, dict)
    assert len(sampling_dictionary) == 5


def test_add_global_variable():
    # Test adding a global variable to a sampling dictionary
    variable_name = "variable_name"
    variable_values = [11, 12, 13, 14, 15]
    sampling_dictionary = {"2019": [1, 2, 3], "2030_min": [6, 7, 8], "2030_max": [9, 10, 11], "2050_min": [12, 13, 14], "2050_max": [15, 16, 17]}
    updated_sampling_dictionary = add_global_variable(variable_name, variable_values, sampling_dictionary)
    
    # Add assertions to check if the global variable is added correctly
    assert isinstance(updated_sampling_dictionary, dict)
    assert len(updated_sampling_dictionary) == 5
    
    # assert that all values in updated_sampling_dictionary have the same length
    assert all(len(updated_sampling_dictionary[key]) == len(list(updated_sampling_dictionary.values())[0]) for key in updated_sampling_dictionary)
    

def test_add_global_variables():
    # Test adding multiple global variables to a sampling dictionary
    dictionary_of_global_variables_to_add = {
        "variable_name_1": [1, 2, 3, 4, 5],
        "variable_name_2": [6, 7, 8, 9, 10],
    }
    dictionary_to_add_to = {"2019": [11, 12, 13], "2030_min": [14, 15, 16], "2030_max": [17, 18, 19], "2050_min": [20, 21, 22], "2050_max": [23, 24, 25]}
    updated_dictionary = add_global_variables(dictionary_of_global_variables_to_add, dictionary_to_add_to)
    
    # Add assertions to check if the global variables are added correctly
    assert isinstance(updated_dictionary, dict)
    assert len(updated_dictionary) == 5
    assert len(updated_dictionary["2019"]) == len(dictionary_to_add_to["2019"]) + len(dictionary_of_global_variables_to_add)

def test_populate_dataframe_interpolated():
    # Test populating a DataFrame with interpolated values
    interpolated_values = {
        "variable_name_1": [1, 2, 3],
        "variable_name_2": [4, 5, 6],
    }
    populated_dataframe = populate_dataframe_interpolated(interpolated_values)
    
    # Add assertions to check if the DataFrame is populated correctly
    assert isinstance(populated_dataframe, dict)
    assert len(populated_dataframe) == 2
    assert "variable_name_1" in populated_dataframe
    assert "variable_name_2" in populated_dataframe
    # assert ...

def test_save_global_parameters_scenarios_as_csv(output_path):
    # Test saving global parameters scenarios as CSV files
    final_scenarios_dfs = {
        "scenario_1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "scenario_2": pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]}),
    }
    save_global_parameters_scenarios_as_csv(final_scenarios_dfs, output_path)
    
    # Add assertions to check if the CSV files are saved correctly
    assert (output_path / "scenario_1.csv").exists()
    assert (output_path / "scenario_2.csv").exists()
    # assert ...

# Add more tests if needed
