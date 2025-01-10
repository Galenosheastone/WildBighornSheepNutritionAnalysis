#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:33:26 2023

@author: galen2
"""
#%% 
#Processing  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'EXAMPLE_DATA_HERE.csv'
data = pd.read_csv(file_path)

# Replace 'ND' with NaN for further processing
data.replace('ND', np.nan, inplace=True)

# Convert the data to numeric, ignoring errors to skip non-numeric columns
data = data.apply(pd.to_numeric, errors='ignore')

# Find 1/5th of the minimum positive values for each column
min_positive_values = data.iloc[:, 2:].apply(lambda x: x[x > 0].min() / 5)

# Replace NaN and 0 with 1/5th of the minimum positive value of the corresponding variable
data.iloc[:, 2:] = data.iloc[:, 2:].apply(lambda x: x.replace(0, np.nan).fillna(min_positive_values[x.name]))

# Perform log transformation on the numeric data, skipping the first two columns
data.iloc[:, 2:] = np.log(data.iloc[:, 2:])

# Autoscaling (mean centering and dividing by the standard deviation)
data_scaled = data.iloc[:, 2:].apply(lambda x: (x - x.mean()) / x.std())

# Recombine the processed data with the first two columns
processed_data = pd.concat([data.iloc[:, :2], data_scaled], axis=1)

# Visualizing some aspects of the processed data
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title("Distribution of a selected variable")
processed_data.iloc[:, 2].hist(bins=30)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.title("Heatmap of processed data")
plt.imshow(processed_data.iloc[:, 2:], aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel("Variable Index")
plt.ylabel("Sample Index")

plt.tight_layout()
plt.show()

# Save the processed data to a new CSV file
processed_file_path = '[INSERT_EXPT]_PROCESSED_DATA_HERE.csv'
processed_data.to_csv(processed_file_path, index=False)

processed_file_path


#%% 
#Visualization of the data
import seaborn as sns

# Loading the original data again for comparison
original_data = pd.read_csv(file_path)
original_data.replace('ND', np.nan, inplace=True)
original_data = original_data.apply(pd.to_numeric, errors='ignore')

# Selecting a subset of columns for visualization (to avoid overcrowding in the plot)
# You can adjust the number of columns based on your preference
selected_columns = original_data.columns[2:12]

# Plotting the boxplots
plt.show()

import random

# To ensure we can select a random set of variables, we need to check the total number of columns available
num_columns = original_data.shape[1] - 2  # Excluding the first two non-numeric columns

# Selecting a random set of 10 columns (if there are at least 10 numeric columns)
if num_columns >= 10:
    random_columns = random.sample(list(original_data.columns[2:]), 10)
else:
    # If less than 10 numeric columns, select all
    random_columns = original_data.columns[2:]

# Plotting the boxplots with the randomly selected columns
plt.figure(figsize=(15, 10))

# Boxplot before processing
plt.subplot(2, 1, 1)
sns.boxplot(data=original_data[random_columns])
plt.title('Concentrations of Random Variables Before Processing')
plt.xticks(rotation=90)

# Boxplot after processing
plt.subplot(2, 1, 2)
sns.boxplot(data=processed_data[random_columns])
plt.title('Concentratons of Random Variables After Processing')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# Generating density plots for the same random set of variables

plt.figure(figsize=(15, 10))

# Density plot before processing
plt.subplot(2, 1, 1)
for col in random_columns:
    sns.kdeplot(original_data[col], label=col, fill=True)
plt.title('Density Plot of Random Variables Before Processing')
plt.xlabel('Concentration')
plt.ylabel('Density')
plt.legend()

# Density plot after processing
plt.subplot(2, 1, 2)
for col in random_columns:
    sns.kdeplot(processed_data[col], label=col, fill=True)
plt.title('Density Plot of Random Variables After Processing')
plt.xlabel('Normalized Concentration')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# Saving the processed data to a CSV file
processed_data_file = 'FINAL_processed_data.csv'
processed_data.to_csv(processed_data_file, index=False)

# Saving the boxplots and density plots
# Re-generating the plots to save them

# Boxplots
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
sns.boxplot(data=original_data[random_columns])
plt.title('Boxplot of Random Variables Before Processing')
plt.xticks(rotation=90)
plt.subplot(2, 1, 2)
sns.boxplot(data=processed_data[random_columns])
plt.title('Boxplot of Random Variables After Processing')
plt.xticks(rotation=90)
plt.tight_layout()
boxplot_file = '_normalization_boxplots.png'
plt.savefig(boxplot_file)
plt.close()

# Density plots
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
for col in random_columns:
    sns.kdeplot(original_data[col], label=col, fill=True)
plt.title('Density Plot of Random Variables Before Processing')
plt.xlabel('Concentration')
plt.ylabel('Density')
plt.legend()
plt.subplot(2, 1, 2)
for col in random_columns:
    sns.kdeplot(processed_data[col], label=col, fill=True)
plt.title('Density Plot of Random Variables After Processing')
plt.xlabel('Normalized Concentration')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
density_plot_file = 'normalization_density_plots.png'
plt.savefig(density_plot_file)
plt.close()

processed_data_file, boxplot_file, density_plot_file

#%%

# Saving the combined density plots to a file

# Re-generating the combined density plots for saving
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
for col in original_data.columns[2:]:
    sns.kdeplot(original_data[col], label='_nolegend_', color='blue', fill=True, alpha=0.1)
plt.title('Combined Density Plot Before Processing')
plt.xlabel('Value')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
for col in processed_data.columns[2:]:
    sns.kdeplot(processed_data[col], label='_nolegend_', color='orange', fill=True, alpha=0.1)
plt.title('Combined Density Plot After Processing')
plt.xlabel('Value')
plt.ylabel('Density')

plt.tight_layout()

# File path for saving
combined_density_plot_file = 'combined_density_plots.png'
plt.savefig(combined_density_plot_file, dpi = 300)
plt.close()

combined_density_plot_file
