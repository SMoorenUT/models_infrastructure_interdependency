
from typing import Union
import warnings
import numpy as np
import pandas as pd
import ema_workbench
import ema_workbench.analysis.scenario_discovery_util as sdutil
import importlib
importlib.reload(sdutil)
importlib.reload(ema_workbench)

def calculate_basic_statistics(data: np.ndarray or list) -> dict: # type: ignore
    if not isinstance(data, (np.ndarray, list)):
        raise TypeError("data must be a numpy array or list")

    statistics = {
        "mean": np.mean(data),
        "median": np.median(data),
        "lower_quartile": np.percentile(data, 25),
        "upper_quartile": np.percentile(data, 75),
        "lower_quintile": np.percentile(data, 20),
        "upper_quintile": np.percentile(data, 80),
        "lower_decile": np.percentile(data, 10),
        "upper_decile": np.percentile(data, 90),
        "minimum": np.min(data),
        "maximum": np.max(data),
        "standard_deviation": np.std(data),
        "variance": np.var(data),
    }

    return statistics


def binarize_array(
    numerical_array: np.array, binarization_threshold: Union[float, int], criterion: str, threshold_is_relative: bool = True
):
    """
    Convert a numerical array to a binary array based on a threshold and criterion.
    Parameters
    ----------
    numerical_array : np.ndarray
        The input array containing numerical values to be binarized.
    binarization_threshold : Union[float, int]
        The threshold value used for binarization. If `threshold_is_relative` is True,
        this should be a float between 0 and 1 representing the percentile. Otherwise, it should be an
        absolute value.
    criterion : str
        The criterion used to compare the numerical array values against the threshold. Must be one of
        the following: '>', '<', '>=', '<=', '=='.
    threshold_is_relative : bool, optional
        If True, the `binarization_threshold` is treated as a percentile (between 0 and 1) and converted
        to an absolute threshold value based on the input array. Default is True.
    Returns
    -------
    np.ndarray
        A binary array where elements are set to 1 if they meet the criterion with respect to the threshold,
        and 0 otherwise.
    Raises
    ------
    TypeError
        If `numerical_array` is not a numpy array, or if `criterion` is not a string, or if `binarization_threshold`
        is not a float or int when `threshold_is_relative` is True.
    ValueError
        If `criterion` is not one of the specified values, or if `binarization_threshold` is not between 0 and 1
        when `threshold_is_relative` is True.
    Notes
    -----
    - The function currently supports the following criteria: '>', '<', '>=', '<=', '=='.
    - A 'between' criterion is planned for future implementation.
    - If `threshold_is_relative` is False and `binarization_threshold` is between 0 and 1,
      a warning is issued but the threshold is not converted.
    Examples
    --------
    >>> import numpy as np
    >>> numerical_array = np.array([1, 2, 3, 4, 5])
    >>> binarize_array(numerical_array, 0.6, '>')
    array([0, 0, 0, 1, 1])
    >>> binarize_array(numerical_array, 3, '==', threshold_is_relative=False)
    array([0, 0, 1, 0, 0])
    Convert the numerical array to a binary array based on the threshold and criterion
    """
    # Check the input types
    criterion = criterion.lower()
    criterion_values = [">", "<", ">=", "<=", "=="]
    # TODO: Implement the 'between' criterion
    if not isinstance(numerical_array, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")
    if not isinstance(criterion, str):
        raise TypeError("criterion must be a string")
    if criterion not in criterion_values:
        raise ValueError(
            f"criterion must be one of the following: {', '.join(criterion_values)}, but criterion is '{criterion}'"
        )

    if threshold_is_relative:
        if not isinstance(binarization_threshold, (float, int)):
            raise TypeError("binarization_threshold must be a float or int")
        if not 0 < binarization_threshold < 1:
            raise ValueError("binarization_threshold must be between 0 and 1 if threshold_is_relative is True")
        absoulute_threshold = np.percentile(numerical_array, binarization_threshold * 100)
    else:
        if 0 < binarization_threshold < 1:
            warnings.warn("The binarization_threshold is between 0 and 1 but not converted to an absolute value as threshold_is_relative is False", UserWarning)
        absoulute_threshold = binarization_threshold
    

    # Convert the numerical array to a binary array based on input threshold and criterion
    if criterion == ">":
        binarized_numerical_array = np.where(numerical_array > absoulute_threshold, 1, 0)
    elif criterion == "<":
        binarized_numerical_array = np.where(numerical_array < absoulute_threshold, 1, 0)
    elif criterion == ">=":
        binarized_numerical_array = np.where(numerical_array >= absoulute_threshold, 1, 0)
    elif criterion == "<=":
        binarized_numerical_array = np.where(numerical_array <= absoulute_threshold, 1, 0)
    elif criterion == "==":
        binarized_numerical_array = np.where(numerical_array == absoulute_threshold, 1, 0)
    else:
        warnings.warn("numerical_array is not converted to a binary array") 
    return binarized_numerical_array


def prim(independent_var_df: pd.DataFrame, dependent_var_array: np.array, binarization_threshold: Union[float, int], criterion: str, threshold_is_relative: bool = True) -> tuple:
    if not isinstance(independent_var_df, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(dependent_var_array, np.ndarray):
        raise TypeError("dependent_var_array must be a numpy array")
  
    dependent_var_array = binarize_array(
        dependent_var_array, binarization_threshold, criterion, threshold_is_relative=True 
    )

    prim_obj = ema_workbench.analysis.prim.Prim(
        independent_var_df,
        dependent_var_array,
        threshold=0.8,
        mode=sdutil.RuleInductionType.BINARY
    )

    box = prim_obj.find_box()

    box.show_tradeoff()

    box.show_pairs_scatter()

    return prim_obj, box

    


def cart(simulation_input: pd.DataFrame, simulation_output: np.array):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")

    result = ema_workbench.analysis.cart.CART(
        simulation_input, simulation_output, mass_min=0.05, mode=sdutil.RuleInductionType.REGRESSION
    )
    return result


def logistic_regression(
    simulation_input: pd.DataFrame, 
    simulation_output: np.array, 
    binarization_threshold: Union[float, int] = None, 
    criterion: str = None, 
    threshold_is_relative: bool = True
):
    if not isinstance(simulation_input, pd.DataFrame):
        raise TypeError("simulation_input must be a pandas DataFrame")
    if not isinstance(simulation_output, np.ndarray):
        raise TypeError("simulation_output must be a numpy array")

    # Convert the continuous output to a binary output if needed
    if binarization_threshold is not None and criterion is not None and not np.all(np.isin(simulation_output, [0, 1])): # Check if the output is already binary and arguments are provided to make binary
        simulation_output = binarize_array(
            numerical_array=simulation_output, 
            binarization_threshold=binarization_threshold, 
            criterion=criterion, 
            threshold_is_relative=threshold_is_relative
        )

    lr_object = ema_workbench.analysis.logistic_regression.Logit(
        simulation_input, simulation_output, threshold=0.95
    )
    lr_object.run(maxiter=50)
    return lr_object