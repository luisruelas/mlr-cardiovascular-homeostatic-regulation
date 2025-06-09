import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class UnivariateAnalysis:
    # Constants as class attributes
    VARIABLES = ['mean_nn', 'sd_nn', 'mean_sbp', 'sd_sbp']
    VARIABLE_MAPPING = {
        'mean_nn': {'full_name': 'Mean NN', 'abv': 'meanIBI'},
        'sd_nn': {'full_name': 'SD NN', 'abv': 'sdIBI'},
        'mean_sbp': {'full_name': 'Mean SBP', 'abv': 'meanSBP'},
        'sd_sbp': {'full_name': 'SD SBP', 'abv': 'sdSBP'},
    }

    def __init__(self, database: str, mnv: float, mhv: float, transform: str = None):
        """
        Initialize UnivariateAnalysis class.
        
        Args:
            database: Database to analyze ('aa' or 'bruno')
            mnv: Maximum normotensive value
            mhv: Minimum hypertensive value
            transform: Data transformation method ('box', 'yeo', or None)
        """
        self.database = database
        self.mnv = mnv
        self.mhv = mhv
        self.transform = transform
        self.data = self._load_data()
        self.data = self._create_bp_population()

    def _transform_data(self) -> pd.DataFrame:
        pass

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

    def perform_anova_analysis(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform one-way ANOVA analysis for each variable across population groups.
        
        Returns:
            Dictionary containing ANOVA results for each bp_population and variable
        """
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        for bp_pop in bp_populations:
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            if len(bp_data) < 2:
                continue
            results[bp_pop] = {}
            
            for var in self.VARIABLES:
                groups = [group[var].values for name, group in bp_data.groupby('population_group')]
                
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    results[bp_pop][var] = {
                        'f_statistic': f_stat,
                        'p_value': p_value
                    }
                else:
                    results[bp_pop][var] = {
                        'f_statistic': None,
                        'p_value': None
                    }
        
        return results
    
    def perform_kruskal_wallis_test(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform Kruskal-Wallis test for each variable across population groups.
        """
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']

        for bp_pop in bp_populations:
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            if len(bp_data) < 2:
                continue
            results[bp_pop] = {}

            for var in self.VARIABLES:
                groups = [group[var].values for name, group in bp_data.groupby('population_group')]
                
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    stat, p_value = stats.kruskal(*groups)
                    results[bp_pop][var] = {
                        'statistic': stat,
                        'p_value': p_value
                    }
                else:
                    results[bp_pop][var] = {
                        'statistic': None,
                        'p_value': None
                    }
        
        return results
    

    def perform_tukey_test(self, variable: str, bp_pop: str) -> pairwise_tukeyhsd:
        """
        Perform Tukey's HSD test for a specific variable and blood pressure population.
        
        Args:
            variable: Variable name to analyze
            bp_pop: Blood pressure population to analyze
            
        Returns:
            Tukey's test results object or None if insufficient data
        """
        bp_data = self.data[self.data['bp_population'] == bp_pop]
        
        if len(bp_data) < 2:
            return None
            
        return pairwise_tukeyhsd(
            endog=bp_data[variable],
            groups=bp_data['population_group'],
            alpha=0.05
        )

    def perform_shapiro_test(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Perform Shapiro-Wilk test for each variable across population groups."""
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        for bp_pop in bp_populations:
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            if len(bp_data) < 2:
                continue
            results[bp_pop] = {}
            
            for var in self.VARIABLES:
                groups = bp_data['population_group'].unique()
                for group in groups:
                    group_data = bp_data[bp_data['population_group'] == group]
                    skew = stats.skew(group_data[var])
                    kurtosis = stats.kurtosis(group_data[var])
                    stat, p_value = stats.shapiro(group_data[var])
                    if results[bp_pop].get(group) is None:
                        results[bp_pop][group] = {}
                    results[bp_pop][group][var] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'skew': skew,
                        'kurtosis': kurtosis
                    }
        return results

    def save_results(self):
        """Save Shapiro-Wilk test results to files."""
        shapiro_results = self.perform_shapiro_test()
        for bp_pop, variables in shapiro_results.items():
            output_file = f'results/normality_tests/shapiro_wilk_{self.database}_{bp_pop}.txt'
            os.makedirs('results/normality_tests', exist_ok=True)
            if os.path.exists(output_file):
                os.remove(output_file)
            for group, variables in shapiro_results[bp_pop].items():
                with open(output_file, 'a') as f:
                    n = len(self.data[self.data['bp_population'] == bp_pop])
                    f.write(f"Shapiro-Wilk Test Results for {bp_pop} - {group}\n")
                    f.write("=" * 50 + "\n\n")
                    for var, stats in variables.items():
                        var_name = self.VARIABLE_MAPPING[var]['full_name']
                        f.write(f"\n{var_name} Analysis:\n")
                        f.write("-" * 30 + "\n")
                        if stats['p_value'] is not None:
                            f.write(f"Shapiro-Wilk and Normality Test Results (n={n}):\n")
                            f.write(f"Skew: {stats['skew']:.4f}\n")
                            f.write(f"Kurtosis: {stats['kurtosis']:.4f}\n")
                            f.write(f"Statistic: {stats['statistic']:.4f}\n")
                            f.write(f"p-value: {stats['p_value']:.4f}\n")
                            if stats['p_value'] < 0.05:
                                f.write("Not normal\n")
                            else:
                                f.write("Normal\n")
                        else:
                            f.write("Insufficient data for analysis\n")
                        f.write("\n" + "=" * 50 + "\n")

        anova_results = self.perform_anova_analysis()
        for bp_pop, variables in anova_results.items():
            n = len(self.data[self.data['bp_population'] == bp_pop])
            output_file = f'results/anova_{self.database}_{bp_pop}.txt'
            os.makedirs('results', exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(f"Statistical Analysis Results for {bp_pop}\n")
                f.write("=" * 50 + "\n\n")
                
                for var, stats in variables.items():
                    var_name = self.VARIABLE_MAPPING[var]['full_name']
                    f.write(f"\n{var_name} Analysis:\n")
                    f.write("-" * 30 + "\n")
                    if stats['p_value'] is not None:
                        f.write(f"ANOVA Results (n={n}):\n")
                        f.write(f"F-statistic: {stats['f_statistic']:.4f}\n")
                        f.write(f"p-value: {stats['p_value']:.4f}\n")
                        
                        if stats['p_value'] < 0.05:
                            f.write("\nTukey's HSD Post-hoc Test:\n")
                            tukey_results = self.perform_tukey_test(var, bp_pop)
                            if tukey_results is not None:
                                f.write(str(tukey_results) + "\n")
                    else:
                        f.write("Insufficient data for analysis\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
