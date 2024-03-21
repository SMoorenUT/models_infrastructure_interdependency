"""
Create scenarios, and global & local variables for the simulation from one file
"""

import numpy as np
import pandas as pd
import datetime as dt
import pathlib
from parameter_generators import parameters_sampling
from scenario_generators import scenario_generator_ema

# Add the path to the current file
CURR_DIR = pathlib.Path(__file__).parent

parameters_sampling.create_scenarios()
scenario_generator_ema.generate_and_output()