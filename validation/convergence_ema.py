#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:16:21 2024

@author: andreavargasf
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Import samples

impacts = pd.DataFrame(pd.read_csv("analysis/ema_road_model_08_05_2024_results.csv",
                                    index_col=0))
sample1 = impacts.iloc[:, -1]
total_samples = len(sample1)
# sample2 = impacts.iloc[:, -1]

#%% Good one

# Function to compute statistics
def compute_statistics(samples):
    mean = np.mean(samples)
    std = np.std(samples)
    return mean, std

# Bootstrap function to calculate confidence intervals
def bootstrap_statistics(samples, increments, B=1000, alpha=0.05):
    means = []
    std_devs = []
    mean_cis = []
    std_cis = []
    sample_sizes = []
    
    for i in increments:
        bootstrap_means = []
        bootstrap_stds = []
        for _ in range(B):
            if len(samples[:i]) > 0:  # To avoid errors with empty sample arrays
                bootstrap_sample = np.random.choice(samples, size=i, replace=False)
                mean, std = compute_statistics(bootstrap_sample)
                bootstrap_means.append(mean)
                bootstrap_stds.append(std)
        
        # Calculate statistics and confidence intervals
        mean = np.percentile(bootstrap_means, 50)
        std = np.percentile(bootstrap_stds, 50)
        mean_ci = np.percentile(bootstrap_means, [100*alpha/2, 100*(1-alpha/2)])
        std_ci = np.percentile(bootstrap_stds, [100*alpha/2, 100*(1-alpha/2)])
        
        means.append(mean)
        std_devs.append(std)
        mean_cis.append(mean_ci)
        std_cis.append(std_ci)
        sample_sizes.append(i)
    
    return means, std_devs, mean_cis, std_cis, sample_sizes

# Samples
samples = sample1
total_samples = len(samples)

# Parameters
step_size = 10
increments = range(step_size, total_samples + 1, step_size)
B = 1000  # Number of bootstrap samples

# Calculate statistics and bootstrap confidence intervals for incremental sample sizes
means, std_devs, mean_cis, std_cis, sample_sizes = bootstrap_statistics(samples, increments, B=B)

# Calculate reference values at the maximum sample size
reference_mean, reference_std = compute_statistics(samples)

# Define tolerance as 2.5% of the reference or current mean/std
tolerance_mean = 0.025 * reference_mean
tolerance_std = 0.025 * reference_std

# Check if confidence intervals are within the tolerance from the reference mean/std and the current mean/std
def within_tolerance(current_value, ci, reference_value, tolerance):
    lower_bound = current_value - tolerance
    upper_bound = current_value + tolerance
    ref_lower_bound = reference_value - tolerance
    ref_upper_bound = reference_value + tolerance
    
    # Check if the confidence interval exceeds either the current value's tolerance or the reference's tolerance
    return (ci[0] >= lower_bound and ci[1] <= upper_bound) and (ci[0] >= ref_lower_bound and ci[1] <= ref_upper_bound)

# Apply this function across all increments to determine the stabilization point
def find_stabilization_point(means, std_devs, mean_cis, std_cis, reference_mean, reference_std):
    consecutive_compliant = 0
    
    for idx in range(len(means)):
        mean_within = within_tolerance(means[idx], mean_cis[idx], reference_mean, tolerance_mean)
        std_within = within_tolerance(std_devs[idx], std_cis[idx], reference_std, tolerance_std)
        
        if mean_within and std_within:
            consecutive_compliant += 1
            # Check if three consecutive points meet the criterion
            if consecutive_compliant >= 3:
                return idx - 1  # return the sample size index
        else:
            consecutive_compliant = 0
    
    return None  # Return None if no stabilization point is found

# Use this function to find the minimum sample size for stabilization
stabilization_point = find_stabilization_point(means, std_devs, mean_cis, std_cis, reference_mean, reference_std)
if stabilization_point is not None:
    print(f"Minimum sample size for stabilization: {sample_sizes[stabilization_point]}")
else:
    print("No stabilization point found within the given tolerances.")
    
# Plotting the results
plt.figure(figsize=(14, 6))

# Plot means with confidence intervals
plt.subplot(1, 2, 1)
plt.axhline(y=reference_mean, color='r', linestyle='--', label=f'Reference Mean at {total_samples}')
plt.plot(sample_sizes, means, label='Mean')
plt.fill_between(sample_sizes, [ci[0] for ci in mean_cis], [ci[1] for ci in mean_cis], color='b', alpha=0.2, label='95% CI')
plt.xlabel('Sample Size')
plt.ylabel('Mean')
plt.title('Convergence of Mean with 95% CI')
plt.grid(True)
plt.legend()

# Plot standard deviations with confidence intervals
plt.subplot(1, 2, 2)
plt.axhline(y=reference_std, color='r', linestyle='--', label=f'Reference Std at {total_samples}')
plt.plot(sample_sizes, std_devs, label='Standard Deviation')
plt.fill_between(sample_sizes, [ci[0] for ci in std_cis], [ci[1] for ci in std_cis], color='orange', alpha=0.2, label='95% CI')
plt.xlabel('Sample Size')
plt.ylabel('Standard Deviation')
plt.title('Convergence of Standard Deviation with 95% CI')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

    
    
    