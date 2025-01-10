#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Oct 1 10:41:666 2024

@author: galen2
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text

# Load your dataset
file_path = 'BHS_FINAL_processed_data.csv'  # Update this path to your file location
data = pd.read_csv(file_path)

# Dynamically identify the first two columns (Sample ID and Group/Treatment)
first_column = data.columns[0]  # Sample ID
second_column = data.columns[1]  # Class/Group/Treatment

# Prepare the data for PCA by excluding the first two columns (non-numeric)
X = data.drop([first_column, second_column], axis=1)

# Custom colors and markers (adjust as needed)
group_colors = {'Early_SM': 'orange', 'Severe_SM': 'blue'}  # Update group names and colors as needed

# Perform PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create 2D PCA data
X_pca_2d = X_pca[:, :2]
pca_df_2d = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
pca_df_2d['Group'] = data[second_column]

# Create 3D PCA data
X_pca_3d = X_pca[:, :3]
pca_df_3d = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['Group'] = data[second_column]
pca_df_3d['color'] = pca_df_3d['Group'].map(group_colors)

# Compute loadings
loadings = pca.components_.T  # shape (n_features, n_components)

# Function to extract top metabolites based on loading magnitudes (PC1 and PC2)
def get_top_metabolites(loadings, top_n=15):
    loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)  # Magnitude for PC1 and PC2
    top_indices = np.argsort(loading_magnitudes)[-top_n:]
    return top_indices

# Extract top metabolite indices based on the PCA loadings (PC1 and PC2)
top_metabolite_indices = get_top_metabolites(loadings, top_n=15)  # You can adjust top_n as needed

# Function to draw a confidence ellipse
def draw_confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
    if len(x) == 0 or len(y) == 0:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_radius_x = np.sqrt(1 + pearson) * n_std
    ellipse_radius_y = np.sqrt(1 - pearson) * n_std
    ellipse = Ellipse((0, 0), width=ellipse_radius_x * 2, height=ellipse_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Enhanced Biplot Function with Non-Overlapping Labels
def biplot_adjusted(pca, X_pca_2d, pca_df_2d, file_name, scale=2, top_metabolite_indices=None, text_color='red'):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    xs = X_pca_2d[:, 0]
    ys = X_pca_2d[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    # Plotting the samples with colors and confidence ellipses
    for group, color in group_colors.items():
        group_data = pca_df_2d[pca_df_2d['Group'] == group]
        plt.scatter(group_data['PC1'] * scalex, group_data['PC2'] * scaley, c=color, label=group, alpha=0.7)
        draw_confidence_ellipse(group_data['PC1'] * scalex, group_data['PC2'] * scaley, ax, facecolor=color, alpha=0.2, edgecolor='black')

    # Plotting the loadings (using the shared top_metabolite_indices)
    loadings = pca.components_.T
    texts = []
    for i in top_metabolite_indices:  # Ensure top metabolites are consistent
        plt.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale, 
                  color='r', alpha=0.5, head_width=0.05, head_length=0.1)
        texts.append(plt.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15, 
                              X.columns[i], color=text_color, ha='center', va='center', fontsize=9, 
                              bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)))

    # Adjust text to prevent overlapping
    adjust_text(texts, 
                only_move={'points':'y', 'texts':'y'},  # Restrict movement to y-axis
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                expand_points=(1.2, 1.2),
                expand_text=(1.2, 1.2),
                force_text=0.5,
                force_points=0.2,
                lim=1000)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
    plt.title('PCA Biplot with Adjusted Labels')
    plt.legend(title='Group', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name, dpi=600)
    plt.show()

# Plotting enhanced biplot with non-overlapping labels
biplot_adjusted(pca, X_pca_2d, pca_df_2d, 'biplot_adjusted.png', scale=2, top_metabolite_indices=top_metabolite_indices, text_color='red')

# Plotting loadings importance plot
def plot_loadings_importance(loadings, top_metabolite_indices, file_name):
    plt.figure(figsize=(10, 8))
    top_loadings = loadings[top_metabolite_indices, :2]
    loading_magnitudes = np.sqrt(top_loadings[:, 0]**2 + top_loadings[:, 1]**2)
    metabolite_names = [X.columns[i] for i in top_metabolite_indices]
    plt.barh(metabolite_names, loading_magnitudes, color='skyblue')
    plt.xlabel('Loading Magnitude')
    plt.ylabel('Metabolites')
    plt.title('Top Metabolites by Loading Magnitude (PC1 and PC2)')
    plt.tight_layout()
    plt.savefig(file_name, dpi=600)
    plt.show()

# Save the loadings importance plot
plot_loadings_importance(loadings, top_metabolite_indices, 'loadings_importance_plot.png')

#%%