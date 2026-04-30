import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats  # Add this import at the top
try:
    from .transformator import Transformator
except ImportError:
    from transformator import Transformator
from matplotlib import colors
import statsmodels.api as sm

class BivariateAnalysis:
    font_size = 18

    # Heatmap style configuration
    HEATMAP_SIG_BORDER_COLOR = 'black'
    HEATMAP_SIG_BORDER_WIDTH = 3
    HEATMAP_R_FONT_SIZE = 16        # r font size for cells >= threshold
    HEATMAP_R_FONT_SIZE_BELOW = 16  # r font size for cells < threshold
    HEATMAP_R_FONT_WEIGHT = 'bold'
    HEATMAP_R_FONT_WEIGHT_BELOW = 'normal'
    HEATMAP_P_FONT_SIZE = 16        # p font size for cells >= threshold
    HEATMAP_P_FONT_SIZE_BELOW = 16   # p font size for cells < threshold
    HEATMAP_P_FONT_WEIGHT = 'bold'
    HEATMAP_P_FONT_WEIGHT_BELOW = 'normal'
    # Reuse the same variables and mapping from UnivariateAnalysis
    VARIABLES = ['mean_nn', 'sd_nn', 'mean_sbp', 'sd_sbp']
    VARIABLE_MAPPING = {
        'mean_nn': {'full_name': 'meanIBI', 'abv': 'meanIBI'},
        'sd_nn': {'full_name': 'sdIBI', 'abv': 'sdIBI'},
        'mean_sbp': {'full_name': 'meanSBP', 'abv': 'meanSBP'},
        'sd_sbp': {'full_name': 'sdSBP', 'abv': 'sdSBP'},
    }
    positions = [0, 0.5, 1]
    cmap_colors = [(103/255, 169/255, 207/255), (247/255, 247/255, 247/255), (239/255, 138/255, 98/255)]
    cmap = colors.LinearSegmentedColormap.from_list('custom_warm', list(zip(positions, cmap_colors)), N=256)

    def __init__(self, database: str, mnv: float, mhv: float, transform: str = None, pearson_r_threshold: float = 0.5):
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
        self.pearson_r_threshold = pearson_r_threshold
        self.data = self._load_data()
        self.data = self._create_bp_population()
        if transform is not None:
            self.data = Transformator.transform_data_by_group(self.data, self.VARIABLES, transform)

    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified database."""
        file_path = ('clean_databases/population_results_autonomic_aging(20yGroups).csv' 
                    if self.database == 'aa' else 'clean_databases/population_results_bruno.csv')
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

    def _annotate_heatmap(self, ax, correlation_matrix, p_values, mask):
        """Draw per-cell r and p annotations with independent styling, and borders on significant cells."""
        mask = np.asarray(mask)
        n = len(self.VARIABLES)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                corr_val = correlation_matrix.iloc[i, j]
                p_val = p_values[i, j]
                p_text = f'{p_val:.4f}' if p_val >= 0.001 else '<0.001'
                cx, cy = j + 0.5, i + 0.5

                cor_meets_threshold = corr_val >= self.pearson_r_threshold
                p_meets_threshold = p_val < 0.05

                r_size = self.HEATMAP_R_FONT_SIZE if cor_meets_threshold else self.HEATMAP_R_FONT_SIZE_BELOW
                r_weight = self.HEATMAP_R_FONT_WEIGHT if cor_meets_threshold else self.HEATMAP_R_FONT_WEIGHT_BELOW
                p_size = self.HEATMAP_P_FONT_SIZE if p_meets_threshold else self.HEATMAP_P_FONT_SIZE_BELOW
                p_weight = self.HEATMAP_P_FONT_WEIGHT if p_meets_threshold else self.HEATMAP_P_FONT_WEIGHT_BELOW


                ax.text(cx, cy - 0.15, f'r={corr_val:.2f}',
                        ha='center', va='center',
                        fontsize=r_size,
                        fontweight=r_weight)
                ax.text(cx, cy + 0.15, f'p={p_text}',
                        ha='center', va='center',
                        fontsize=p_size,
                        fontweight=p_weight)

                if cor_meets_threshold and p_meets_threshold:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor=self.HEATMAP_SIG_BORDER_COLOR,
                        linewidth=self.HEATMAP_SIG_BORDER_WIDTH,
                        zorder=3,
                        clip_on=False,
                    ))

    def create_simple_regression_plots(self):
        """Create simple regression plots for each blood pressure population."""
        # use log transformation for the variables
        population_groups = self.data['population_group'].unique()
        os.makedirs('results/simple_regression_plots', exist_ok=True)
        for population_group in population_groups:
            already_done_plots = []
            # Filter data for current bp population
            population_group_data = self.data[self.data['population_group'] == population_group]
            if len(population_group_data) == 0:
                continue
            for variable in self.VARIABLES:
                for other_variable in self.VARIABLES:
                    if variable != other_variable:
                        already_done = False
                        for plot in already_done_plots:
                            if variable in plot and other_variable in plot:
                                already_done = True
                                break
                        if already_done:
                            continue
                        done_plots_text = f'{variable}_vs_{other_variable}'
                        already_done_plots.append(done_plots_text)
                        # also plot the linear regression line
                        X = population_group_data[other_variable]
                        Y = population_group_data[variable]
                        X = sm.add_constant(X)
                        model = sm.OLS(Y, X).fit()
                        predictions = model.predict(X)
                        plt.plot(X[other_variable], predictions, color='black', linewidth=2)
                        plt.scatter(population_group_data[other_variable], population_group_data[variable])
                        plt.xlabel(other_variable)
                        plt.ylabel(variable)
                        plt.title(f'{variable} vs {other_variable} - {population_group}')
                        plt.savefig(f'results/simple_regression_plots/{self.database}_{population_group}_{variable}_vs_{other_variable}_simple_regression.png')
                        plt.close()

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
                
                # Create mask for non-significant correlations (p > 0.05) and r is less than threshold and diagonal
                mask = np.eye(n, dtype=bool)
                # mask = (p_values > 0.05) | (abs(correlation_matrix) < self.pearson_r_threshold) | np.eye(n, dtype=bool)
                
                # Set non-significant correlations and diagonal to NaN (will appear white)
                correlation_matrix_masked = correlation_matrix.copy()
                correlation_matrix_masked[mask] = 0

                plt.figure(figsize=(10, 8))

                ax = sns.heatmap(
                    correlation_matrix_masked,
                    annot=False,
                    cmap=self.cmap,
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,
                    fmt='',
                    linewidths=2,
                    linecolor='lightgray',
                    cbar=False,
                )
                self._annotate_heatmap(ax, correlation_matrix, p_values, mask)

                plt.xticks(fontsize=self.font_size)
                plt.yticks(fontsize=self.font_size)
                
                # Use full variable names for labels
                labels = [self.VARIABLE_MAPPING[var]['full_name'] for var in self.VARIABLES]
                plt.xticks(np.arange(len(labels)) + 0.5, labels)
                plt.yticks(np.arange(len(labels)) + 0.5, labels)
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
                mask = (p_values > 0.05) | (abs(correlation_matrix) < self.pearson_r_threshold) | np.eye(n, dtype=bool)
                correlation_matrix_masked = correlation_matrix.copy()
                correlation_matrix_masked[mask] = 0

                plt.figure(figsize=(10, 8))

                ax = sns.heatmap(
                    correlation_matrix_masked,
                    annot=False,
                    cmap=self.cmap,
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,
                    fmt='',
                    linewidths=2,
                    linecolor='lightgray',
                    cbar=False,
                )
                self._annotate_heatmap(ax, correlation_matrix, p_values, mask)

                plt.xticks(fontsize=self.font_size)
                plt.yticks(fontsize=self.font_size)
                # Use full variable names for labels
                labels = [self.VARIABLE_MAPPING[var]['full_name'] for var in self.VARIABLES]
                plt.xticks(np.arange(len(labels)) + 0.5, labels)
                plt.yticks(np.arange(len(labels)) + 0.5, labels)
                
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