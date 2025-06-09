import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats  # Add this import at the top

class BivariateAnalysis:
    # Reuse the same variables and mapping from UnivariateAnalysis
    VARIABLES = ['mean_nn', 'sd_nn', 'mean_sbp', 'sd_sbp']
    VARIABLE_MAPPING = {
        'mean_nn': {'full_name': 'Mean NN', 'abv': 'meanIBI'},
        'sd_nn': {'full_name': 'SD NN', 'abv': 'sdIBI'},
        'mean_sbp': {'full_name': 'Mean SBP', 'abv': 'meanSBP'},
        'sd_sbp': {'full_name': 'SD SBP', 'abv': 'sdSBP'},
    }

    def __init__(self, database: str, mnv: float, mhv: float):
        """
        Initialize BivariateAnalysis class.
        
        Args:
            database: Database to analyze ('aa' or 'bruno')
            mnv: Maximum normotensive value
            mhv: Minimum hypertensive value
        """
        self.database = database
        self.mnv = mnv
        self.mhv = mhv
        self.data = self._load_data()
        self.data = self._create_bp_population()

    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified database."""
        file_path = ('data/clean_databases/population_results_autonomic_aging(20yGroups).csv' 
                    if self.database == 'aa' else 'data/clean_databases/population_results_bruno.csv')
        return pd.read_csv(file_path)

    def _create_bp_population(self) -> pd.DataFrame:
        """Create blood pressure population column based on SBP values."""
        conditions = [
            (self.data['mean_sbp'] <= self.mnv),
            (self.data['mean_sbp'] > self.mnv) & (self.data['mean_sbp'] < self.mhv),
            (self.data['mean_sbp'] >= self.mhv)
        ]
        values = ['normal_bp', 'intermediate_bp', 'high_bp']
        self.data['bp_population'] = np.select(conditions, values, default='unknown')
        return self.data

    def create_pearson_correlation_heatmaps(self):
        """Create and save correlation heatmaps for each blood pressure population."""
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        # Create results directory if it doesn't exist
        os.makedirs('results/correlation_heatmaps', exist_ok=True)
        
        for bp_pop in bp_populations:
            # Filter data for current bp population
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            
            if len(bp_data) < 2:
                continue
            
            population_groups = bp_data.population_group.unique()
            
            for population_group in population_groups:
                # Filter data for current population group
                group_data = bp_data[bp_data['population_group'] == population_group]
                
                if len(group_data) < 2:  # Skip if not enough data points
                    continue
                
                correlation_matrix = group_data[self.VARIABLES].corr(method='pearson')
                
                # Calculate p-values matrix
                p_values = np.zeros_like(correlation_matrix)
                n = len(self.VARIABLES)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            stat, p = stats.pearsonr(group_data[self.VARIABLES[i]], group_data[self.VARIABLES[j]])
                            p_values[i, j] = p
                        else:
                            p_values[i, j] = 0  # p-value for self correlation
                
                # Create mask for non-significant correlations (p > 0.05) and diagonal
                mask = (p_values > 0.05) | np.eye(n, dtype=bool)
                
                # Set non-significant correlations and diagonal to NaN (will appear white)
                correlation_matrix_masked = correlation_matrix.copy()
                correlation_matrix_masked[mask] = np.nan

                plt.figure(figsize=(10, 8))
                
                # Create a custom annotation array that combines correlation and p-values
                annot_matrix = np.empty((n, n), dtype=object)
                for i in range(n):
                    for j in range(n):
                        corr_val = correlation_matrix.iloc[i, j]
                        p_val = p_values[i, j]
                        if i == j:  # If on diagonal
                            annot_matrix[i, j] = ''  # Empty annotation for diagonal
                        elif p_val <= 0.05:  # If significant
                            annot_matrix[i, j] = f'r={corr_val:.2f}\np={p_val:.10f}'
                        else:
                            annot_matrix[i, j] = f'ns\np={p_val:.10f}'
                
                sns.heatmap(
                    correlation_matrix_masked,
                    annot=annot_matrix,  # Use our custom annotation matrix
                    cmap='coolwarm',  # Color scheme
                    vmin=-1, vmax=1,  # Correlation range
                    center=0,  # Center the colormap at 0
                    square=True,  # Make cells square
                    fmt='',  # Empty format since we're using custom annotations
                    mask=None,  # Don't mask any values
                    annot_kws={'size': 8},  # Reduced size to fit both values
                    linewidths=2,  # Add grid lines with width 2
                    linecolor='lightgray'  # Set grid color to light gray
                )
                
                # Use full variable names for labels
                labels = [self.VARIABLE_MAPPING[var]['full_name'] for var in self.VARIABLES]
                plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
                plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
                
                plt.title(f'Pearson Correlation Heatmap\n{bp_pop.replace("_", " ").title()} - {population_group} n={len(group_data)}\n* ns = non-significant (p > 0.05)')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(f'results/correlation_heatmaps/{self.database}_{bp_pop}_{population_group}_pearson_correlation.png')
                plt.close()
                
                # Save numerical results with both correlation and p-values
                with open(f'results/correlation_heatmaps/{self.database}_{bp_pop}_{population_group}_pearson_correlation.txt', 'w') as f:
                    f.write(f"Correlation Analysis Results for {bp_pop} - {population_group}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("Correlation Matrix:\n")
                    f.write(str(correlation_matrix.round(10)) + "\n\n")  # Show full precision
                    f.write("P-values Matrix:\n")
                    f.write(str(pd.DataFrame(p_values, columns=self.VARIABLES, index=self.VARIABLES).round(10)))  # Show full precision

    def create_spearman_correlation_heatmaps(self):
        """Create and save correlation heatmaps for each blood pressure population."""
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        # Create results directory if it doesn't exist
        os.makedirs('results/correlation_heatmaps', exist_ok=True)
        
        for bp_pop in bp_populations:
            # Filter data for current bp population
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            
            if len(bp_data) < 2:
                continue
            
            population_groups = bp_data.population_group.unique()
            
            for population_group in population_groups:
                # Filter data for current population group
                group_data = bp_data[bp_data['population_group'] == population_group]
                
                if len(group_data) < 2:  # Skip if not enough data points
                    continue
                
                correlation_matrix = group_data[self.VARIABLES].corr(method='spearman')
                
                # Calculate p-values matrix
                p_values = np.zeros_like(correlation_matrix)
                n = len(self.VARIABLES)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            stat, p = stats.spearmanr(group_data[self.VARIABLES[i]], group_data[self.VARIABLES[j]])
                            p_values[i, j] = p
                        else:
                            p_values[i, j] = 0  # p-value for self correlation
                
                # Create mask for non-significant correlations (p > 0.05) and diagonal
                mask = (p_values > 0.05) | np.eye(n, dtype=bool)
                
                # Set non-significant correlations and diagonal to NaN (will appear white)
                correlation_matrix_masked = correlation_matrix.copy()
                correlation_matrix_masked[mask] = np.nan

                plt.figure(figsize=(10, 8))
                
                # Create a custom annotation array that combines correlation and p-values
                annot_matrix = np.empty((n, n), dtype=object)
                for i in range(n):
                    for j in range(n):
                        corr_val = correlation_matrix.iloc[i, j]
                        p_val = p_values[i, j]
                        if i == j:  # If on diagonal
                            annot_matrix[i, j] = ''  # Empty annotation for diagonal
                        elif p_val <= 0.05:  # If significant
                            annot_matrix[i, j] = f'r={corr_val:.2f}\np={p_val:.10f}'
                        else:
                            annot_matrix[i, j] = f'ns\np={p_val:.10f}'
                
                sns.heatmap(
                    correlation_matrix_masked,
                    annot=annot_matrix,  # Use our custom annotation matrix
                    cmap='coolwarm',  # Color scheme
                    vmin=-1, vmax=1,  # Correlation range
                    center=0,  # Center the colormap at 0
                    square=True,  # Make cells square
                    fmt='',  # Empty format since we're using custom annotations
                    mask=None,  # Don't mask any values
                    annot_kws={'size': 8},  # Reduced size to fit both values
                    linewidths=2,  # Add grid lines with width 2
                    linecolor='lightgray'  # Set grid color to light gray
                )
                
                # Use full variable names for labels
                labels = [self.VARIABLE_MAPPING[var]['full_name'] for var in self.VARIABLES]
                plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
                plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
                
                plt.title(f'Spearman Correlation Heatmap\n{bp_pop.replace("_", " ").title()} - {population_group} n={len(group_data)}\n* ns = non-significant (p > 0.05)')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(f'results/correlation_heatmaps/{self.database}_{bp_pop}_{population_group}_spearman_correlation.png')
                plt.close()
                
                # Save numerical results with both correlation and p-values
                with open(f'results/correlation_heatmaps/{self.database}_{bp_pop}_{population_group}_spearman_correlation.txt', 'w') as f:
                    f.write(f"Correlation Analysis Results for {bp_pop} - {population_group}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("Correlation Matrix:\n")
                    f.write(str(correlation_matrix.round(10)) + "\n\n")  # Show full precision
                    f.write("P-values Matrix:\n")
                    f.write(str(pd.DataFrame(p_values, columns=self.VARIABLES, index=self.VARIABLES).round(10)))  # Show full precision