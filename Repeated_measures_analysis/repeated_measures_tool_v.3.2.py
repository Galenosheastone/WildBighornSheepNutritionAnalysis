#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sept 25 13:33:15 2024

@author: galen2
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# Set Seaborn style
sns.set(style="whitegrid")

# Set global font size for all plots
plt.rcParams.update({
    'axes.titlesize': 18,  # Title size
    'axes.labelsize': 16,  # Axis labels size
    'xtick.labelsize': 14,  # X-axis tick size
    'ytick.labelsize': 14,  # Y-axis tick size
    'legend.fontsize': 14,  # Legend font size
    'text.color': 'black'   # Text color (if needed for visibility)
})

# Load the dataset
file_path = 'Repeated_measures_v2.0.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Handling "ND" values and converting metabolite concentrations to numeric
data_cleaned = data.replace("ND", np.nan)
data_cleaned.iloc[:, 3:] = data_cleaned.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')

# Randomly select one sample for each animal with a fixed seed for reproducibility
np.random.seed(42)
random_selected_samples = data_cleaned.groupby('Animal Id').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
random_selected_samples_numeric = random_selected_samples.apply(pd.to_numeric, errors='coerce')

# Standardize the metabolite concentration data
scaler = StandardScaler()
data_scaled_original = scaler.fit_transform(data_cleaned.iloc[:, 3:])  # Exclude the first three columns (Sample Number, Animal Id, Nutritional State)
data_scaled_modified = scaler.fit_transform(random_selected_samples_numeric.iloc[:, 3:])

# Impute missing values with the mean for each column in both datasets
for i in range(data_scaled_original.shape[1]):
    col_mean_original = np.nanmean(data_scaled_original[:, i])
    col_mean_modified = np.nanmean(data_scaled_modified[:, i])
    data_scaled_original[np.isnan(data_scaled_original[:, i]), i] = col_mean_original
    data_scaled_modified[np.isnan(data_scaled_modified[:, i]), i] = col_mean_modified

# Perform PCA and get explained variance for Original dataset
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result_original = pca.fit_transform(data_scaled_original)
explained_variance_original = pca.explained_variance_ratio_

pca_df_original = pd.DataFrame(data=pca_result_original, columns=['PC1', 'PC2'])
pca_df_original['Nutritional State'] = data_cleaned['Nutritional State'].values.astype(str)  # Ensure 'Nutritional State' is a string

# Perform PCA and get explained variance for Modified dataset
pca_result_modified = pca.fit_transform(data_scaled_modified)
explained_variance_modified = pca.explained_variance_ratio_

pca_df_modified = pd.DataFrame(data=pca_result_modified, columns=['PC1', 'PC2'])
pca_df_modified['Nutritional State'] = random_selected_samples['Nutritional State'].values.astype(str)  # Use the original non-numeric DataFrame

# Set up the figure layout
fig = plt.figure(figsize=(20, 20))  # Adjusted figure size
grid_spec = fig.add_gridspec(nrows=5, ncols=6)  # 5x6 grid (1 row for PCA, 3x3 for histograms, each spanning two columns)

# Identify all unique 'Nutritional State' categories across both datasets
nutritional_states = sorted(set(pca_df_original['Nutritional State'].unique()) | set(pca_df_modified['Nutritional State'].unique()))

# Create a consistent color palette mapping
palette = sns.color_palette("tab10", n_colors=len(nutritional_states))
color_mapping = dict(zip(nutritional_states, palette))

# PCA plots side by side, each taking three columns (ensuring equal size)
ax_pca_original = fig.add_subplot(grid_spec[0, :3])  # First three columns for Original PCA
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Nutritional State',
    data=pca_df_original,
    palette=color_mapping,
    hue_order=nutritional_states,
    ax=ax_pca_original,
    legend=False,
    alpha=0.8
)
ax_pca_original.set_title('Original Dataset PCA')
ax_pca_original.set_xlabel(f'PC1 ({explained_variance_original[0] * 100:.2f}% Variance)')
ax_pca_original.set_ylabel(f'PC2 ({explained_variance_original[1] * 100:.2f}% Variance)')

ax_pca_modified = fig.add_subplot(grid_spec[0, 3:])  # Last three columns for Modified PCA
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Nutritional State',
    data=pca_df_modified,
    palette=color_mapping,
    hue_order=nutritional_states,
    ax=ax_pca_modified,
    legend='brief',
    alpha=0.8
)
ax_pca_modified.set_title('Modified Dataset PCA')
ax_pca_modified.set_xlabel(f'PC1 ({explained_variance_modified[0] * 100:.2f}% Variance)')
ax_pca_modified.set_ylabel(f'PC2 ({explained_variance_modified[1] * 100:.2f}% Variance)')

np.random.seed(666)

# Randomly select 9 metabolites for the distribution analysis (to fit in a 3x3 grid)
metabolite_columns = data_cleaned.columns[3:]  # Exclude the first three columns (Sample Number, Animal Id, Nutritional State)
selected_metabolites = np.random.choice(metabolite_columns, size=12, replace=True)

# Create a 3x3 grid of histograms, each spanning two columns like the PCA plots
for idx, metabolite in enumerate(selected_metabolites):
    row = 1 + idx // 3  # Start at row 1 for histograms
    col_start = (idx % 3) * 2  # Use two columns for each histogram
    ax = fig.add_subplot(grid_spec[row, col_start:col_start+2])
    
    # Plot the histograms for original and modified datasets
    sns.histplot(data_cleaned[metabolite], kde=True, color='blue', label='Original', bins=30, stat='density', alpha=0.6, ax=ax)
    sns.histplot(random_selected_samples_numeric[metabolite], kde=True, color='orange', label='Modified', bins=30, stat='density', alpha=0.6, ax=ax)
    
    # Calculate KS statistic
    ks_statistic, ks_p_value = ks_2samp(
        data_cleaned[metabolite].dropna(),
        random_selected_samples_numeric[metabolite].dropna()
    )
    
    # Set the title of the plot and annotate with the KS statistic and p-value
    ax.set_title(f'{metabolite}', fontsize=16)  # Adjust title font size
    ax.set_xlabel('')
    ax.set_ylabel('Density')
    
    # Add the KS statistic and p-value to the plot at the center right
    text_str = f'KS Stat: {ks_statistic:.2f}\nP-Value: {ks_p_value:.2e}'
    ax.text(0.95, 0.5, text_str, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
    
    # Add the legend only to the first plot
    if idx == 0:
        ax.legend()

# Adjust the layout and save the final plot
plt.tight_layout()
plt.savefig('PCA_and_Histograms_with_KS_Statistics_Center_Right.png', dpi=600)
plt.close()

print("Combined PCA and histograms with KS statistics centered right saved as 'PCA_and_Histograms_with_KS_Statistics_Center_Right.png'.")

#%%
import pandas as pd

# Load the dataset
df = pd.read_csv('Repeated_measures_test_v1.0.csv')

# Assuming 'Sample_ID' is the column that identifies individual samples, and 'Timepoint' (or similar) identifies repeated measures.
# Modify these column names as per your dataset

# Group by Sample_ID and count the occurrences to identify repeated measures
summary = df.groupby('Animal Id').size().reset_index(name='Count')

# Filter to show only those samples with more than one measure
repeated_measures_summary = summary[summary['Count'] > 1]

# Get the detailed information for where repeated measures happened
detailed_repeats = df[df['Animal Id'].isin(repeated_measures_summary['Animal Id'])]

# Save the summary and detailed information to CSV files
repeated_measures_summary.to_csv('repeated_measures_summary.csv', index=False)
detailed_repeats.to_csv('detailed_repeated_measures.csv', index=False)

# Print the summary and detailed information
print(repeated_measures_summary)  # For Spyder or similar environments
print(detailed_repeats)#%%


