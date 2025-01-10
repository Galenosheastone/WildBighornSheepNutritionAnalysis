#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 2024-10-07 15:13:01.454843

@author: galen2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as mt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import datetime

x = datetime.datetime.now()
print(x)

# Load the dataset
file_path = 'FINAL_processed_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)
# Prepare data for analysis
groups = data['Group'].unique()
metabolites = data.columns[2:]

# Perform Mann-Whitney U test
results = []
for metabolite in metabolites:
    group_data = [data[data['Group'] == group][metabolite] for group in groups]
    stat, p = mannwhitneyu(*group_data)
    results.append([metabolite, stat, p])

results_df = pd.DataFrame(results, columns=['Metabolite', 'Statistic', 'P-value'])

# Apply FDR correction
results_df['Adjusted P-value'] = mt.multipletests(results_df['P-value'], method='fdr_bh')[1]

# Filter for significant results and sort
significant_metabolites = results_df[results_df['Adjusted P-value'] < 0.05].sort_values(by='Adjusted P-value')

# Melt the dataframe for easier plotting
melted_data = pd.melt(data, id_vars=['Group'], value_vars=significant_metabolites['Metabolite'], var_name='Metabolite', value_name='Level')

# Define custom colors for the boxplots
custom_palette = {group: color for group, color in zip(groups, ["#f2a918", "#1a1aac"])}

# Create the boxplot
fig, ax1 = plt.subplots(figsize=(20, 12))
sns.boxplot(x='Metabolite', y='Level', hue='Group', data=melted_data, palette=custom_palette, order=significant_metabolites['Metabolite'], ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=55, fontsize=18)
ax1.set_xlabel('', fontsize=20)
ax1.set_ylabel('Level', fontsize=22)
ax1.set_ylim(-4, 4)  # Adjust these values based on your data
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(fontsize=18, title_fontsize=20)

# Add vertical lines between metabolites
num_metabolites = len(significant_metabolites)
for i in range(num_metabolites - 1):
    ax1.axvline(i + 0.5, color='gray', linestyle='--', linewidth=0.7)

# Normalize p-values for coloring
norm = mcolors.Normalize(vmin=significant_metabolites['Adjusted P-value'].min(), vmax=0.05)

# Create a custom colormap from light blue to dark blue
light_blue = "#add8e6"
dark_blue = "#00008b"
medium_blue = "#1a1aac"

cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", [light_blue, dark_blue])

# Create a ScalarMappable with the custom colormap
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Adjust color bar to represent p-value significance with the new colormap
cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04, aspect=50, location='top')
cbar.set_label('Adjusted P-value Significance', labelpad=10, fontsize=20)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()

# Save the figure as a PNG file
output_file_path = 'MannWhitU_figure_ordered_with_gradient_here.png'  # Adjust to your desired file path
plt.savefig(output_file_path, format='png', dpi=1000)

plt.show()