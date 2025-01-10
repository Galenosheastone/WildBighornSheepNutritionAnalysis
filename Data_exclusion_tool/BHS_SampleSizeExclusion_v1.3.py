#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:03:34 2024

@author: galen2
"""

#%%
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skbio.stats.distance import permanova
from skbio.stats.distance import DistanceMatrix
from scipy.spatial.distance import pdist, squareform

# Load the dataset
file_path = 'Helo_captures_early_mod_severe.csv'
data = pd.read_csv(file_path)

# Replace 'ND' (Non-Detects) with NaN
data.replace('ND', np.nan, inplace=True)

# Remove rows with missing or NaN data
data_cleaned = data.dropna()

# Identify the metabolite columns (everything after the first two columns)
metabolite_columns = data_cleaned.columns[2:]  # Metabolite data starts from the 3rd column

# 1. ANOVA Function to compare groups
def perform_anova(data, groups_column, metabolite_columns):
    anova_results = {}
    for metabolite in metabolite_columns:
        # Convert the metabolite data to float for ANOVA
        groups = [data[data[groups_column] == group][metabolite].astype(float) 
                  for group in data[groups_column].unique()]
        anova_results[metabolite] = stats.f_oneway(*groups)
    # Filter significant results (p < 0.05)
    significant_results = {k: v for k, v in anova_results.items() if v.pvalue < 0.05}
    return anova_results, significant_results

# Perform ANOVA with all groups
anova_results_all, significant_results_all = perform_anova(data_cleaned, 'Nurtitional State', metabolite_columns)

# Save ANOVA results to CSV
anova_df_all = pd.DataFrame(anova_results_all).T
anova_df_all.columns = ['F-Statistic', 'p-value']
anova_df_all.to_csv('anova_results_with_all_groups.csv', index=True)

# Exclude "Mod-SM" group
data_no_mod = data_cleaned[data_cleaned['Nurtitional State'] != 'Mod-SM']

# Perform ANOVA excluding "Mod-SM"
anova_results_no_mod, significant_results_no_mod = perform_anova(data_no_mod, 'Nurtitional State', metabolite_columns)

# Save ANOVA results excluding "Mod-SM" to CSV
anova_df_no_mod = pd.DataFrame(anova_results_no_mod).T
anova_df_no_mod.columns = ['F-Statistic', 'p-value']
anova_df_no_mod.to_csv('anova_results_without_mod_group.csv', index=True)

# 2. PCA Analysis and visualization with variance displayed
def plot_pca(data, metabolite_columns, group_column, title, filename, dpi=100, custom_palette='Set1'):
    pca_data = data[metabolite_columns].astype(float)
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(pca_data)
    
    # Extract the explained variance ratios
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame with PCA results and group info
    pca_df = pd.DataFrame(pca_transformed, columns=['PC1', 'PC2'])
    pca_df['Group'] = data[group_column]
    
    # Plot PCA results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', palette=custom_palette)
    plt.title(title)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)')
    plt.savefig(filename, dpi=dpi)
    plt.show()

# Plot PCA with all groups and save
plot_pca(data_cleaned, metabolite_columns, 'Nurtitional State', 'PCA with All Groups', 'pca_with_all_groups.png', dpi=600, custom_palette='viridis')

# Plot PCA without "Mod-SM" and save
plot_pca(data_no_mod, metabolite_columns, 'Nurtitional State', 'PCA without Mod-SM Group', 'pca_without_mod_group.png', dpi=600, custom_palette='plasma')

# 3. KS Test for distribution comparison
def ks_test_distribution(data_full, data_excluded, metabolite_columns):
    ks_results = {}
    
    for metabolite in metabolite_columns:
        # Perform KS test between the full dataset and the excluded dataset for each metabolite
        full_values = data_full[metabolite].astype(float)
        excluded_values = data_excluded[metabolite].astype(float)
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(full_values, excluded_values)
        ks_results[metabolite] = {'KS Statistic': ks_stat, 'p-value': p_value}
    
    return ks_results

# Compare the full data (with Mod-SM) and the data without Mod-SM using KS test
ks_results = ks_test_distribution(data_cleaned, data_no_mod, metabolite_columns)

# Save KS test results to CSV
ks_df = pd.DataFrame(ks_results).T
ks_df.to_csv('ks_test_results.csv', index=True)

# 4. PERMANOVA Analysis
def perform_permanova(data, group_column, metabolite_columns, metric='euclidean'):
    # Extract the metabolite data and group labels
    metabolite_data = data[metabolite_columns].astype(float).values
    group_labels = data[group_column].values
    
    # Calculate the distance matrix (e.g., using Euclidean or Bray-Curtis distance)
    distance_matrix = pdist(metabolite_data, metric=metric)
    distance_matrix = squareform(distance_matrix)  # Convert to a square format
    
    # Create a scikit-bio DistanceMatrix object
    dist_matrix = DistanceMatrix(distance_matrix, ids=data.index)
    
    # Perform PERMANOVA
    permanova_results = permanova(dist_matrix, group_labels)
    
    return permanova_results

# Perform PERMANOVA with all groups using Euclidean distance
permanova_results_all = perform_permanova(data_cleaned, 'Nurtitional State', metabolite_columns)

# Exclude "Mod-SM" group and perform PERMANOVA again
permanova_results_no_mod = perform_permanova(data_no_mod, 'Nurtitional State', metabolite_columns)

# Save PERMANOVA results to text files
with open('permanova_results_with_all_groups.txt', 'w') as f:
    f.write(str(permanova_results_all))

with open('permanova_results_without_mod_group.txt', 'w') as f:
    f.write(str(permanova_results_no_mod))

# Print the results
print("PERMANOVA results with all groups:")
print(permanova_results_all)

print("\nPERMANOVA results without 'Mod-SM' group:")
print(permanova_results_no_mod)

#%%