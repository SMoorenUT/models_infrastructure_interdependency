import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file
df = pd.read_csv(os.path.join(CURR_DIR, "ema_road_model_08_05_2024_results.csv"), index_col=0)

# Compute the covariance matrix (only for numeric columns)
cov_matrix = df.cov()

# Print the covariance matrix
print(cov_matrix)