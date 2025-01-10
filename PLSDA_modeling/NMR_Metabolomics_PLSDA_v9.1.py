#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:24:34 2024

@author: galen2
"""

#%%

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import permutation_test

# Load the dataset
file_path = 'FINAL_processed_data.csv'
nmr_data = pd.read_csv(file_path)

# Preparing the dataset for PLSDA
X = nmr_data.drop(columns=["Sample Number", "Group"])
y = nmr_data["Group"]

# Encode the categorical variable 'Group'
y_encoded = pd.factorize(y)[0]

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=6)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine the optimal number of components
def optimize_components(X_train, y_train):
    max_components = min(X_train.shape)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mean_r2 = []

    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        scores = cross_val_predict(pls, X_train, y_train, cv=kf)
        r2 = r2_score(y_train, scores)
        mean_r2.append(r2)

    optimal_components = np.argmax(mean_r2) + 1
    return optimal_components

optimal_components = optimize_components(X_train_scaled, y_train)
plsda = PLSRegression(n_components=optimal_components)
plsda.fit(X_train_scaled, y_train)

#PLSDA modeling is complete here, below is all viz and eval 

# Permutation Test for model validation with visualization including p-value
def perform_permutation_test_with_visualization(model, X, y, n_permutations=1000):
    original_score = r2_score(y, model.predict(X))
    perm_scores = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm_y = np.random.permutation(y)
        perm_score = r2_score(perm_y, model.predict(X))
        perm_scores[i] = perm_score

    p_value = np.mean(perm_scores > original_score)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(perm_scores, kde=True, color='blue', label='Permutation R² Scores')
    plt.axvline(x=original_score, color='red', linestyle='--', label=f'Original R² Score: {original_score:.2f}')
    plt.title('Permutation Test R² Score Distribution')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')

    # Display the p-value on the plot
    plt.text(x=original_score, y=plt.ylim()[1]*0.9, s=f'p-value: {p_value:.4f}', color='red', ha='right')

    plt.show()

    return p_value

p_value = perform_permutation_test_with_visualization(plsda, X_train_scaled, y_train)
print(f"Permutation Test p-value: {p_value}")

# Perform predictions and model evaluation

# Calculate VIP scores
def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_  # Scores
    w = pls_model.x_weights_  # Weights
    q = pls_model.y_loadings_  # Y Loadings

    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j]) for j in range(h)])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)

    return vips

vip_scores = calculate_vip_scores(plsda, X_train_scaled, y_train)


#%%

# Sort VIP scores and select the top 15 variables
sorted_indices = np.argsort(vip_scores)[::-1][:15]
top_vip_features = np.array(X.columns)[sorted_indices]
top_vip_scores = vip_scores[sorted_indices]

# Filter the dataset to include only the top VIP features and the 'Group' column
filtered_vip_data = nmr_data[['Group'] + list(top_vip_features)]

# Pivoting the data for the heatmap
pivot_vip_data = filtered_vip_data.melt(id_vars=['Group'], var_name='Metabolite', value_name='Concentration')
sorted_heatmap_data = pivot_vip_data.pivot_table(index='Metabolite', columns='Group', values='Concentration').loc[top_vip_features]

# Visualization: Combined Bar Plot and Heatmap
blue_gold_palette = mcolors.LinearSegmentedColormap.from_list("BlueGold", ["blue", "gold"])
colors = [blue_gold_palette(i/15) for i in range(15)]

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.10])
#adjusts the width ratios to help with the heatmap

ax0 = fig.add_subplot(gs[0])
sns.barplot(x=top_vip_scores, y=top_vip_features, palette=colors, ax=ax0)
ax0.set_title('Top 15 Features in PLSDA Model by VIP Scores')
ax0.set_xlabel('VIP Scores')
ax0.set_ylabel('Metabolites')

ax1 = fig.add_subplot(gs[1])
sns.heatmap(sorted_heatmap_data, annot=False, cmap=blue_gold_palette, linewidths=.5, ax=ax1)
ax1.set_title('Metabolite Concentrations')
ax1.set_xlabel('')
ax1.set_ylabel('Metabolite')
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('blue_gold_combined_visualization_plsda_HERE.png', dpi=300)
plt.show()

# Predictions and model performance
y_pred_train = plsda.predict(X_train_scaled)
y_pred_test = plsda.predict(X_test_scaled)

# Convert predictions to binary
y_pred_train_binary = np.where(y_pred_train > 0.5, 1, 0)
y_pred_test_binary = np.where(y_pred_test > 0.5, 1, 0)

# Compute metrics
conf_matrix = confusion_matrix(y_test, y_pred_test_binary)
roc_auc = roc_auc_score(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Visualization
# Plotting confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax[0].text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
ax[0].set_xlabel('Predictions')
ax[0].set_ylabel('Actuals')
ax[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
ax[1].plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
ax[1].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
ax[1].legend()

plt.tight_layout()
plt.show()

# Calculating Q2 and R2 for the model
def calculate_q2_r2(y_true, y_pred):
    ss_total = sum((y_true - np.mean(y_true)) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total
    q2 = 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)
    return q2, r2

q2_train, r2_train = calculate_q2_r2(y_train, y_pred_train)
q2_test, r2_test = calculate_q2_r2(y_test, y_pred_test)

# Update visualization for Q2 and R2 with total values
fig, ax = plt.subplots(figsize=(8, 5))
scores = [q2_train, q2_test, r2_train, r2_test]
labels = ['Train Q2', 'Test Q2', 'Train R2', 'Test R2']
colors = ['red', 'orange', 'yellow', 'green']

ax.bar(labels, scores, color=colors)
for i, score in enumerate(scores):
    ax.text(i, score, f'{score:.2f}', ha='center', va='bottom')

ax.set_title('Q2 and R2 Scores')
ax.set_ylabel('Score')
plt.show()

# Cross-validation for model stability

# Number of folds
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare data
X_scaled = scaler.fit_transform(X)
y_encoded = pd.factorize(y)[0]

# Initialize a list to store Q2 values for each fold
q2_scores = []

for train_index, test_index in kf.split(X_scaled):
    # Splitting the data
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Training the model
    plsda = PLSRegression(n_components=5)
    plsda.fit(X_train, y_train)
    
    # Making predictions
    y_pred = plsda.predict(X_test)

    # Calculating Q2
    q2, _ = calculate_q2_r2(y_test, y_pred)
    q2_scores.append(q2)
    
# Calculating the average Q2 score
avg_q2_score = np.mean(q2_scores)
print(f"Average cross validated (10 fold) Q2 Score: {avg_q2_score:.2f}")

# Plotting Q2 scores vs Number of Components
n_components_range = range(1, 16)  # Adjust as needed
avg_q2_scores = []

for n_components in n_components_range:
    fold_q2_scores = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        plsda = PLSRegression(n_components=n_components)
        plsda.fit(X_train, y_train)
        y_pred = plsda.predict(X_test)

        q2 = calculate_q2_r2(y_test, y_pred)
        fold_q2_scores.append(q2)

    avg_q2 = np.mean(fold_q2_scores)
    avg_q2_scores.append(avg_q2)

# Identify the optimal number of components
optimal_components = np.argmax(avg_q2_scores) + 1
print(f"Optimal number of components: {optimal_components}")

# Plotting
fig, ax = plt.subplots()
ax.plot(n_components_range, avg_q2_scores, label='Q2', marker='o')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Q2 Score')
ax.set_title('Q2 Scores vs Number of Components')
plt.show()


def save_summary_to_file(filename):
    summary = []

    # Add summary of the dataset
    summary.append("Dataset Summary:")
    summary.append(f"Number of Samples: {nmr_data.shape[0]}")
    summary.append(f"Number of Features: {nmr_data.shape[1]}")
    summary.append("")

    # Add summary of the model training
    summary.append("Model Training Summary:")
    summary.append(f"Optimal Number of Components: {optimal_components}")
    summary.append("")

    # Add summary of the model validation
    summary.append("Model Validation Summary:")
    summary.append(f"Permutation Test p-value: {p_value}")
    summary.append("")

    # Add summary of VIP scores
    summary.append("VIP Scores Summary:")
    for metabolite, score in zip(top_vip_features, top_vip_scores):
        summary.append(f"{metabolite}: {score}")
    summary.append("")

    # Add summary of model evaluation
    summary.append("Model Evaluation Summary:")
    summary.append(f"Training Q2 Score: {q2_train}")
    summary.append(f"Training R2 Score: {r2_train}")
    summary.append(f"Test Q2 Score: {q2_test}")
    summary.append(f"Test R2 Score: {r2_test}")
    summary.append(f"Confusion Matrix:\n{conf_matrix}")
    summary.append(f"ROC AUC Score: {roc_auc}")
    summary.append("")

    # Add summary of cross-validation
    summary.append("Cross-Validation Summary:")
    summary.append(f"Average Q2 Score (10-fold): {avg_q2_score}")
    summary.append(f"Optimal Number of Components (based on CV): {optimal_components}")
    summary.append("")

    # Save to file
    with open(filename, 'w') as file:
        file.write('\n'.join(summary))

# Save the summary to a text file
save_summary_to_file('analysis_summary.txt')



#%%


import matplotlib.pyplot as plt

# Create a figure with a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

# VIP Scores Bar Chart
axes[0, 0].bar(top_vip_features, top_vip_scores)
axes[0, 0].set_title('Top VIP Scores')
axes[0, 0].set_ylabel('VIP Score')
axes[0, 0].set_xticklabels(top_vip_features, rotation=45, ha='right')

# Heatmap for Feature Importance
sns.heatmap(sorted_heatmap_data, annot=True, cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Feature Importance Heatmap')

# Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')

# ROC Curve
axes[1, 1].plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
axes[1, 1].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()

# Text Box for Key Metrics
metrics_text = f"Permutation Test p-value: {p_value}\nTraining Q2 Score: {q2_train}\nTraining R2 Score: {r2_train}\nTest Q2 Score: {q2_test}\nTest R2 Score: {r2_test}\nOptimal Components: {optimal_components}"
axes[2, 0].axis('off')
axes[2, 0].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12)

# Hide the last subplot (if not needed)
axes[2, 1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()


def create_summary(filename='nmr_analysis_summary.txt'):
    with open(filename, 'w') as file:
        # Summary of the dataset
        file.write("Dataset Summary:\n")
        file.write(f"Number of Samples: {nmr_data.shape[0]}\n")
        file.write(f"Number of Features: {nmr_data.shape[1]}\n\n")

        # Summary of model performance
        file.write("Model Performance Summary:\n")
        file.write(f"Optimal Number of Components: {optimal_components}\n")
        file.write(f"Permutation Test p-value: {p_value:.4f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")

        # Summary of VIP scores
        file.write("VIP Scores for Top Features:\n")
        for feature, score in zip(top_vip_features, top_vip_scores):
            file.write(f"{feature}: {score:.4f}\n")
        file.write("\n")

        # Summary of predictions and model evaluation
        file.write("Model Evaluation:\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n")
        file.write(f"Train Q2 Score: {q2_train:.4f}\n")
        file.write(f"Test Q2 Score: {q2_test:.4f}\n")
        file.write(f"Train R2 Score: {r2_train:.4f}\n")
        file.write(f"Test R2 Score: {r2_test:.4f}\n\n")

        # Summary of Cross-Validation
        file.write("Cross-Validation Summary:\n")
        file.write(f"Average Q2 Score (10-fold CV): {avg_q2_score:.4f}\n")
        file.write(f"Optimal Number of Components (based on CV): {optimal_components}\n")

    print(f"Summary saved to {filename}")

# Call the function to create and save the summary
create_summary()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Sort VIP scores and features in ascending order
sorted_indices = np.argsort(top_vip_scores)
sorted_vip_scores = top_vip_scores[sorted_indices]
sorted_vip_features = top_vip_features[sorted_indices]

# Variables to adjust sizes
lollipop_marker_size = 10   # Size of the markers in the lollipop plot
heatmap_fontsize = 18    # Font size for the heatmap labels
barplot_fontsize = 18    # Font size for the barplot labels
title_fontsize = 20      # Font size for plot titles
label_fontsize = 18      # Font size for axis labels

# Create a figure with gridspec
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # Adjusted width ratios for a narrower heatmap

# Lollipop plot for VIP Scores (Horizontal, ascending order)
ax0 = fig.add_subplot(gs[0])
ax0.hlines(y=range(len(sorted_vip_features)), xmin=0, xmax=sorted_vip_scores, color='skyblue')
ax0.plot(sorted_vip_scores, range(len(sorted_vip_features)), "D", markersize=lollipop_marker_size)  # Adjust marker size
ax0.set_title('VIP Scores', fontsize=title_fontsize)
ax0.set_xlabel('VIP Score', fontsize=label_fontsize)
ax0.set_yticks(range(len(sorted_vip_features)))
ax0.set_yticklabels(sorted_vip_features, fontsize=heatmap_fontsize)  # Adjust y-tick label size

# Heatmap for Feature Importance (Narrower, without cell values)
ax1 = fig.add_subplot(gs[1])
sns.heatmap(sorted_heatmap_data, annot=False, cmap='coolwarm', ax=ax1)  # Set annot to False
ax1.set_title('Feature Importance', fontsize=title_fontsize)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, fontsize=heatmap_fontsize)  # Adjust x-tick label size
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=heatmap_fontsize)  # Adjust y-tick label size
ax1.figure.tight_layout()  # Tight layout for heatmap

plt.tight_layout()

# Save the figure
plt.savefig('combined_vips_heatmap_horizontal_customizable.png', dpi=600)

# Display the plot
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to calculate the explained variance for each component in PLS
def calculate_explained_variance(X, scores):
    """
    Calculate the explained variance for each component in PLS.
    :param X: Standardized original data.
    :param scores: Scores from the PLS model.
    :return: Array of variance explained by each component.
    """
    total_variance = np.sum(X**2)
    explained_variance = np.sum(scores**2, axis=0) / total_variance
    return explained_variance

from matplotlib.patches import Ellipse

def plot_2d_scores(scores, variance_ratio, labels, colors, group_names, 
                  x_label='PLS1', y_label='PLS2', title='',
                  point_size=10, marker_style='o', save_path=None, 
                  title_fontsize=20, label_fontsize=16, tick_fontsize=14, legend_fontsize=12):
    """
    Plot a 2D scores plot for the PLSDA model with customizable font sizes and shaded 95% confidence intervals.
    
    Parameters:
    - scores: ndarray, shape (n_samples, n_components)
        Scores from the PLSDA model.
    - variance_ratio: list or array-like, length >= 2
        Variance explained by each component.
    - labels: array-like, shape (n_samples,)
        Labels for each data point.
    - colors: dict
        Dictionary mapping labels to colors.
    - group_names: dict
        Dictionary mapping labels to group names for the legend.
    - x_label: str, optional (default='PLS1')
        Label for the x-axis.
    - y_label: str, optional (default='PLS2')
        Label for the y-axis.
    - title: str, optional (default='')
        Title of the plot.
    - point_size: int or array-like, optional (default=10)
        Size of the points.
    - marker_style: str, optional (default='o')
        Marker style for the points.
    - save_path: str, optional (default=None)
        Path to save the plot image.
    - title_fontsize: int, optional (default=20)
        Font size for the plot title.
    - label_fontsize: int, optional (default=16)
        Font size for the axis labels.
    - tick_fontsize: int, optional (default=14)
        Font size for the tick labels.
    - legend_fontsize: int, optional (default=12)
        Font size for the legend.
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 8))
    
    for label in unique_labels:
        label_scores = scores[labels == label, :2]
        mean_x, mean_y = np.mean(label_scores, axis=0)
        std_x, std_y = np.std(label_scores, axis=0)
        
        # Scatter plot for each group
        plt.scatter(label_scores[:, 0], label_scores[:, 1], 
                    color=colors[label], s=point_size, marker=marker_style, 
                    label=group_names[label], alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # Adding a shaded ellipse for 95% confidence interval
        ellipse_width = 2 * 1.96 * std_x
        ellipse_height = 2 * 1.96 * std_y
        ellipse = Ellipse((mean_x, mean_y), ellipse_width, ellipse_height, 
                          edgecolor=colors[label], facecolor=colors[label], 
                          alpha=0.2, linewidth=2)
        plt.gca().add_patch(ellipse)
    
    plt.xlabel(f'{x_label} ({variance_ratio[0]*100:.2f}% variance explained)', fontsize=label_fontsize)
    plt.ylabel(f'{y_label} ({variance_ratio[1]*100:.2f}% variance explained)', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, title='Groups')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust tick label sizes
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
    
    plt.show()
    
# Calculating explained variance
scores = plsda.x_scores_
explained_variance = calculate_explained_variance(X_train_scaled, scores)

# Define your colors for each group (modify as needed)
colors = {0: 'Orange', 1: 'blue'}  # Example color mapping

# Define custom group names (modify as needed)
group_names = {0: 'Early-SM', 1: 'Severe-SM'}

# Define the save file path
save_file_path = '2D_PLSDA_plot.png'  # Replace with your desired file path

# Example usage with new parameters for point size and marker style
plot_2d_scores(scores, explained_variance, y_train, colors, group_names, point_size=100, marker_style='*', save_path=save_file_path)

#%%this is where 3D plotting exists 
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scores(scores, variance_ratio, labels, colors, group_names, x_label='PLS1', y_label='PLS2', z_label='PLS3', 
                   title='', point_size=10, marker_style='o', save_path=None, 
                   title_fontsize=20, label_fontsize=16, tick_fontsize=14, legend_fontsize=12, elev=40, azim=100):
    """
    Plot a 3D scores plot for the PLSDA model with customizable font sizes.
    
    Parameters:
    - title_fontsize: Font size for the plot title.
    - label_fontsize: Font size for the axis labels.
    - tick_fontsize: Font size for the tick labels.
    - legend_fontsize: Font size for the legend.
    """
    fig = plt.figure(figsize=(14, 14))  # Adjusted size for better visibility
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_scores = scores[labels == label, :3]  # Ensure scores have at least three components
        ax.scatter(label_scores[:, 0], label_scores[:, 1], label_scores[:, 2], 
                   color=colors[label], s=point_size, marker=marker_style, 
                   label=group_names[label])

    ax.set_xlabel(f'{x_label} ({variance_ratio[0]*100:.2f}% variance explained)', fontsize=label_fontsize)
    ax.set_ylabel(f'{y_label} ({variance_ratio[1]*100:.2f}% variance explained)', fontsize=label_fontsize)
    ax.set_zlabel(f'{z_label} ({variance_ratio[2]*100:.2f}% variance explained)', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True)
    
    # Adjust tick label sizes
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.tick_params(axis='z', labelsize=tick_fontsize)

    ax.view_init(elev=elev, azim=azim)

    if save_path:
        plt.savefig(save_path, format='png', dpi=1000, bbox_inches='tight')

    plt.show()


# Example usage of the 3D plotting function with custom view angles
plot_3d_scores(scores, explained_variance, y_train, colors, group_names, 
               point_size=50, marker_style='*', save_path='3D_PLSDA_plot.png', elev=30, azim=120)

#%%
#%% Loadings Plots

# 2D Loadings Plot
def plot_2d_loadings(pls_model, feature_names, x_label='PLS1', y_label='PLS2', title='2D Loadings Plot'):
    loadings = pls_model.x_loadings_
    plt.figure(figsize=(10, 8))
    plt.scatter(loadings[:, 0], loadings[:, 1], color='blue')
    for i, feature in enumerate(feature_names):
        plt.text(loadings[i, 0], loadings[i, 1], feature, fontsize=9)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.show()

# 3D Loadings Plot
def plot_3d_loadings(pls_model, feature_names, x_label='PLS1', y_label='PLS2', z_label='PLS3', title='3D Loadings Plot'):
    loadings = pls_model.x_loadings_
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2], color='blue')
    for i, feature in enumerate(feature_names):
        ax.text(loadings[i, 0], loadings[i, 1], loadings[i, 2], feature, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.grid(True)
    plt.show()

# Plot the 2D Loadings Plot
plot_2d_loadings(plsda, X.columns)

# Plot the 3D Loadings Plot
plot_3d_loadings(plsda, X.columns)




#%% Enhanced Loadings Plots with Customizable Colors, Labeling, and Group Labels

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Function to calculate the magnitude of the loadings
def calculate_magnitude(loadings):
    return np.sqrt(np.sum(loadings**2, axis=1))

# Function to get the dominant group for each feature
def get_dominant_group(pls_model, X, y):
    loadings = pls_model.x_loadings_
    magnitudes = calculate_magnitude(loadings)
    dominant_groups = []
    for i in range(loadings.shape[0]):
        feature_data = X.iloc[:, i]
        group_means = [feature_data[y == group].mean() for group in np.unique(y)]
        dominant_group = np.argmax(group_means)
        dominant_groups.append(dominant_group)
    return dominant_groups, magnitudes

dominant_groups, magnitudes = get_dominant_group(plsda, X, y)

# 2D Enhanced Loadings Plot
def plot_2d_loadings_enhanced(pls_model, feature_names, dominant_groups, magnitudes, top_n=None, 
                              group_colors=None, group_labels=None, num_labeled=None, x_label='PLS1', y_label='PLS2', title='2D Loadings Plot'):
    loadings = pls_model.x_loadings_
    plt.figure(figsize=(12, 10))

    # Set up color mapping
    unique_groups = np.unique(dominant_groups)
    if group_colors is None:
        cmap = cm.get_cmap('Set1', len(unique_groups))
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}
    else:
        group_colors = {group: group_colors[i] for i, group in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(loadings[:, 0], loadings[:, 1], c=[group_colors[group] for group in dominant_groups], 
                    s=magnitudes*200, alpha=0.6)

    # Determine the indices of the top N metabolites
    if top_n:
        top_indices = np.argsort(magnitudes)[-top_n:]
    else:
        top_indices = range(len(feature_names))

    # Determine the indices of the metabolites to be labeled
    if num_labeled:
        labeled_indices = np.argsort(magnitudes)[-num_labeled:]
    else:
        labeled_indices = top_indices

    for i in labeled_indices:
        ax.text(loadings[i, 0], loadings[i, 1], feature_names[i], fontsize=9, ha='right')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.6) 
               for color in group_colors.values()]
    if group_labels is None:
        labels = [f'Group {group}' for group in unique_groups]
    else:
        labels = [group_labels[group] for group in unique_groups]
    ax.legend(handles, labels, title="Group", loc='best')

    plt.show()

# 3D Enhanced Loadings Plot
def plot_3d_loadings_enhanced(pls_model, feature_names, dominant_groups, magnitudes, top_n=None, 
                              group_colors=None, group_labels=None, num_labeled=None, x_label='PLS1', y_label='PLS2', z_label='PLS3', title='3D Loadings Plot'):
    loadings = pls_model.x_loadings_
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set up color mapping
    unique_groups = np.unique(dominant_groups)
    if group_colors is None:
        cmap = cm.get_cmap('Set1', len(unique_groups))
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}
    else:
        group_colors = {group: group_colors[i] for i, group in enumerate(unique_groups)}

    sc = ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2], c=[group_colors[group] for group in dominant_groups], 
                    s=magnitudes*200, alpha=0.6)

    # Determine the indices of the top N metabolites
    if top_n:
        top_indices = np.argsort(magnitudes)[-top_n:]
    else:
        top_indices = range(len(feature_names))

    # Determine the indices of the metabolites to be labeled
    if num_labeled:
        labeled_indices = np.argsort(magnitudes)[-num_labeled:]
    else:
        labeled_indices = top_indices

    for i in labeled_indices:
        ax.text(loadings[i, 0], loadings[i, 1], loadings[i, 2], feature_names[i], fontsize=9, ha='right')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.grid(True)

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.6) 
               for color in group_colors.values()]
    if group_labels is None:
        labels = [f'Group {group}' for group in unique_groups]
    else:
        labels = [group_labels[group] for group in unique_groups]
    ax.legend(handles, labels, title="Group", loc='best')

    plt.show()

# Define custom colors and labels for the groups (example: 2 groups with custom colors and labels)
custom_group_colors = ['orange', 'blue']
custom_group_labels = ['Early-SM', 'Severe-SM']

# Plot the 2D Enhanced Loadings Plot, showing only the top 10 significant metabolites and labeling the top 5
plot_2d_loadings_enhanced(plsda, X.columns, dominant_groups, magnitudes, top_n=10, group_colors=custom_group_colors, group_labels=custom_group_labels, num_labeled=45)

# Plot the 3D Enhanced Loadings Plot, showing only the top 10 significant metabolites and labeling the top 5
plot_3d_loadings_enhanced(plsda, X.columns, dominant_groups, magnitudes, top_n=10, group_colors=custom_group_colors, group_labels=custom_group_labels, num_labeled=45)


#%% Biplots with Enhanced Visibility and Adjustable Labeling

from mpl_toolkits.mplot3d import Axes3D

# 2D Biplot
def plot_2d_biplot(pls_model, X, y, feature_names, group_colors=None, group_labels=None, num_labeled=10,
                   x_label='PLS1', y_label='PLS2', title='2D Biplot', arrow_scale=20, arrow_color='red'):
    scores = pls_model.x_scores_
    loadings = pls_model.x_loadings_
    plt.figure(figsize=(12, 10))
    
    # Set up color mapping
    unique_groups = np.unique(y)
    if group_colors is None:
        cmap = cm.get_cmap('Set1', len(unique_groups))
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}
    else:
        group_colors = {group: group_colors[i] for i, group in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the scores
    for group in unique_groups:
        ax.scatter(scores[y == group, 0], scores[y == group, 1], label=group_labels[group] if group_labels else f'Group {group}',
                   color=group_colors[group], alpha=0.6)

    # Plot the loadings as arrows
    for i, (loading_x, loading_y) in enumerate(loadings[:, :2]):
        ax.arrow(0, 0, loading_x * arrow_scale, loading_y * arrow_scale, color=arrow_color, alpha=0.8, head_width=0.05, head_length=0.1)
        if num_labeled is None or i in np.argsort(np.linalg.norm(loadings[:, :2], axis=1))[-num_labeled:]:
            ax.text(loading_x * arrow_scale, loading_y * arrow_scale, feature_names[i], color='black', ha='center', va='center', fontsize=9)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    
    ax.legend(title="Group", loc='best')
    plt.show()

# 3D Biplot
def plot_3d_biplot(pls_model, X, y, feature_names, group_colors=None, group_labels=None, num_labeled=20,
                   x_label='PLS1', y_label='PLS2', z_label='PLS3', title='3D Biplot', arrow_scale=20, arrow_color='red'):
    scores = pls_model.x_scores_
    loadings = pls_model.x_loadings_
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up color mapping
    unique_groups = np.unique(y)
    if group_colors is None:
        cmap = cm.get_cmap('Set1', len(unique_groups))
        group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}
    else:
        group_colors = {group: group_colors[i] for i, group in enumerate(unique_groups)}

    # Plot the scores
    for group in unique_groups:
        ax.scatter(scores[y == group, 0], scores[y == group, 1], scores[y == group, 2], label=group_labels[group] if group_labels else f'Group {group}',
                   color=group_colors[group], alpha=0.6)

    # Plot the loadings as arrows
    for i, (loading_x, loading_y, loading_z) in enumerate(loadings[:, :3]):
        ax.quiver(0, 0, 0, loading_x * arrow_scale, loading_y * arrow_scale, loading_z * arrow_scale, color=arrow_color, alpha=0.8, arrow_length_ratio=0.1)
        if num_labeled is None or i in np.argsort(np.linalg.norm(loadings[:, :3], axis=1))[-num_labeled:]:
            ax.text(loading_x * arrow_scale, loading_y * arrow_scale, loading_z * arrow_scale, feature_names[i], color='black', ha='center', va='center', fontsize=9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.grid(True)
    
    ax.legend(title="Group", loc='best')
    plt.show()

# Define custom colors and labels for the groups (example: 2 groups with custom colors and labels)
custom_group_colors = ['orange', 'blue']
custom_group_labels = ['Early-SM', 'Severe-SM']

# Plot the 2D Biplot with custom parameters
plot_2d_biplot(plsda, X_train_scaled, y_train, X.columns, group_colors=custom_group_colors, group_labels=custom_group_labels, num_labeled=5)

# Plot the 3D Biplot with custom parameters
plot_3d_biplot(plsda, X_train_scaled, y_train, X.columns, group_colors=custom_group_colors, group_labels=custom_group_labels, num_labeled=5)
#%%




