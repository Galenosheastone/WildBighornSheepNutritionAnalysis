# WildBighornSheepNutritionAnalysis
# README for Bighorn Sheep Metabolomics Analysis Repository

## Overview
This repository contains Python scripts and datasets used for statistical analysis in the publication of a paper on the metabolic profiles of wild bighorn sheep under different nutritional states. These scripts facilitate various preprocessing, visualization, and statistical analysis tasks, including repeated measures analysis, PCA, PLSDA, two-way ANOVA, and non-parametric tests. The datasets are used to support and validate these analyses.

## Files and Descriptions

### Scripts

1. **BHS_SampleSizeExclusion_v1.3.py**
   - **Purpose:** Performs statistical analysis to evaluate the impact of excluding groups with small sample sizes. Includes comparisons of distributions, visualizations, and key statistical tests.
   - **Key Features:**
     - Generates PCA plots and histograms to visualize the impact of sample size exclusion.
     - Compares distributions using KS statistics.

2. **NMR_Metabolomics_MannWhitneyU_FDR_v4.2.py**
   - **Purpose:** Conducts non-parametric Mann-Whitney U tests for group comparisons and adjusts p-values using the False Discovery Rate (FDR) method.
   - **Key Features:**
     - Identifies significant metabolites between treatment groups.
     - Outputs results in tabular format for further interpretation.

3. **NMR_Metabolomics_PCA_v6.1.py**
   - **Purpose:** Performs Principal Component Analysis (PCA) on NMR metabolomics data.
   - **Key Features:**
     - Generates 2D and 3D PCA plots with confidence ellipsoids.
     - Highlights significant metabolite vectors based on criteria.
     - Provides preprocessing steps for normalization and scaling.

4. **NMR_Metabolomics_PLSDA_v9.1.py**
   - **Purpose:** Implements Partial Least Squares Discriminant Analysis (PLSDA) for classifying samples based on metabolomics data.
   - **Key Features:**
     - Calculates VIP scores.
     - Generates visualizations such as bar plots and heatmaps.
     - Includes cross-validation and permutation tests.

5. **NMR_metabolomics_PROCESSING_MAIN_v2.2.py**
   - **Purpose:** Preprocesses raw NMR metabolomics data for downstream analysis.
   - **Key Features:**
     - Handles imputation for missing and zero values.
     - Performs log transformation and autoscaling.
     - Generates before-and-after visualizations of normalization.

6. **NMR_metabolomics_2wayANOVA_2factor_v2.7.py**
   - **Purpose:** Conducts two-way ANOVA to analyze the effects of two factors and their interaction on metabolite levels.
   - **Key Features:**
     - Processes multifactorial metabolomics data.
     - Adjusts p-values using FDR correction.
     - Visualizes top significant metabolites with customized boxplots.
     - Outputs results for significant metabolites.

7. **repeated_measures_tool_v.3.2.py**
   - **Purpose:** Analyzes repeated measures data to assess longitudinal changes.
   - **Key Features:**
     - Fits mixed-effects models.
     - Visualizes trends over time with confidence intervals.

### Datasets

1. **FINAL_processed_data.csv**
   - **Description:** Fully processed dataset used in final analyses. Contains metabolite concentrations and metadata such as treatment group and sample ID.

2. **MAIN_DATASET_Helo_captures_early_mod_severe.csv**
   - **Description:** Main dataset containing metabolomics data for early, moderate, and severe nutritional states. Includes metadata for statistical grouping.

3. **multifactor_processed_data.csv**
   - **Description:** Processed dataset for multifactor analysis, including several experimental factors and corresponding metabolite data.

4. **Repeated_measures_v2.0.xlsx**
   - **Description:** Repeated measures dataset, organized by individual sample IDs with longitudinal data for statistical analysis.

## Installation and Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bighorn-sheep-metabolomics.git
   ```
2. Install the required dependencies (listed in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Python version 3.9 or later is installed.

## Usage Instructions

1. **Data Preprocessing:**
   - Use `NMR_metabolomics_PROCESSING_MAIN_v2.2.py` to preprocess raw data for downstream analysis.

2. **Exploratory Analysis:**
   - Run `NMR_Metabolomics_PCA_v6.1.py` to perform PCA and generate visualizations.

3. **Statistical Tests:**
   - Use `NMR_Metabolomics_MannWhitneyU_FDR_v4.2.py` for group comparisons and significance testing.
   - Execute `NMR_metabolomics_2wayANOVA_2factor_v2.7.py` for two-way ANOVA and interaction effects.

4. **Classification:**
   - Execute `NMR_Metabolomics_PLSDA_v9.1.py` for PLSDA analysis and group classification.

5. **Longitudinal Analysis:**
   - Run `repeated_measures_tool_v.3.2.py` to analyze repeated measures data and visualize trends.

6. **Sample Size Impact:**
   - Use `BHS_SampleSizeExclusion_v1.3.py` to assess the effect of excluding small groups from the analysis.

## Contribution Guidelines
Contributions are welcome! Please submit a pull request or open an issue for any suggested improvements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, please contact [your email address].

