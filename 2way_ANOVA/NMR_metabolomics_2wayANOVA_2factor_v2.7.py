#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 15:24:27 2024

@author: galen2
"""
#%%

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.stats.multicomp as multi
import scikit_posthocs as sp

# Load the dataset
file_path = 'multifactor_processed_data.csv'  # Update this path if needed
data = pd.read_csv(file_path)

# Replace spaces in column names with underscores for formula compatibility
data.columns = data.columns.str.replace(' ', '_')
print("Adjusted column names:", data.columns)  # Confirm column names

# Correct the misspelled column name 'Nurtitional_State'
data.rename(columns={'Nurtitional_State': 'Nutritional_State'}, inplace=True)

# Ensure categorical variables are correctly defined
if 'Environment' in data.columns and 'Nutritional_State' in data.columns:
    data['Environment'] = pd.Categorical(data['Environment'])
    data['Nutritional_State'] = pd.Categorical(data['Nutritional_State'])
else:
    raise KeyError("Required columns 'Environment' or 'Nutritional_State' not found in dataset.")

# Prepare for ANOVA
metabolites = data.columns[3:]  # Update index as necessary to include all metabolite columns
anova_results = {}

# Perform two-way ANOVA for each metabolite
for metabolite in metabolites:
    formula = f'Q("{metabolite}") ~ C(Environment) + C(Nutritional_State) + C(Environment):C(Nutritional_State)'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[metabolite] = anova_table

# Convert ANOVA results to DataFrame and save
anova_full_results_df = pd.concat(anova_results, axis=0)
anova_full_results_df.index.names = ['Metabolite', 'Source']
anova_full_results_df.reset_index(inplace=True)
anova_full_results_df.to_csv('anova_full_results.csv', index=False)

# Extract p-values and adjust
anova_pvals = {metabolite: anova_table["PR(>F)"].values[:3] for metabolite, anova_table in anova_results.items()}
anova_pvals_df = pd.DataFrame(anova_pvals).T
anova_pvals_df.columns = ['Environment_p', 'NutritionalState_p', 'Interaction_p']
adjusted_pvals = multipletests(anova_pvals_df.values.flatten(), method='fdr_bh')[1]
anova_pvals_adjusted_df = pd.DataFrame(adjusted_pvals.reshape(anova_pvals_df.shape), index=anova_pvals_df.index, columns=anova_pvals_df.columns)

# Replace non-significant values with 'NS'
anova_pvals_adjusted_df = anova_pvals_adjusted_df.applymap(lambda x: 'NS' if x >= 0.05 else x)

significant_results = anova_pvals_adjusted_df[anova_pvals_adjusted_df != 'NS']

# Number of top metabolites to visualize based on interaction effects
top_n = 12  # Select the top metabolites
top_metabolites = significant_results.dropna(subset=['Interaction_p']).sort_values(by='Interaction_p').head(top_n).index.tolist()

# Set up the matplotlib figure and grid
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 6)  # Adjust grid size based on the number of top metabolites

# Loop through the top metabolites and create a boxplot for each
for i, metabolite in enumerate(top_metabolites):
    ax = fig.add_subplot(gs[i])
    
    # Plotting the boxplot with enhanced visibility
    custom_palette = {  # You can define custom colors for each nutritional state here
        'Low': 'lightblue',
        'Medium': 'coral',
        'High': 'green',
        'Early_SM': 'goldenrod',  # Added missing level to avoid KeyError
        'Severe_SM': 'mediumblue'     # Added additional level to handle potential cases
    }
    boxplot = sns.boxplot(x='Environment', y=metabolite, hue='Nutritional_State', data=data, ax=ax,
                          palette=custom_palette, flierprops=dict(marker='d', markerfacecolor='grey', markersize=6, linestyle='none', markeredgecolor='none'), linewidth=2.5)

    ax.set_title(metabolite, fontsize=18)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # To avoid clutter, only display legend for the first plot
    if i > 0:
        ax.legend([],[], frameon=False)
    else:
        ax.legend(fontsize=14)

plt.tight_layout()
plt.suptitle('Boxplots of Top Significant Metabolites by Interaction', fontsize=20, y=1.05)
plt.savefig('top_significant_metabolites_interaction_boxplots_customized.png', dpi=1000)  # Added dpi for resolution control
plt.show()

#%%

# Assuming top_metabolites and significant_results are already defined

# Filter the DataFrame for only top metabolites
top_metabolites_data = anova_pvals_adjusted_df.loc[top_metabolites]

# Export to CSV
top_metabolites_data.to_csv('top_n_significant_metabolites.csv', index=True)

print(f'The top {len(top_metabolites)} significant metabolites have been exported to "top_n_significant_metabolites.csv".')

#%%
