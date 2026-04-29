import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from .transformator import Transformator
import seaborn as sns

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
        self.data_transformed_without_group = self._load_data()
        self.data_transformed_without_group = self._create_bp_population()
        self.should_divide_transformation_by_group = True
        if transform is not None:
            self.untransformed_data = self.data_transformed_without_group.copy()
            self.data_transformed_with_group = Transformator.transform_data_by_group(self.data_transformed_without_group, self.VARIABLES, self.transform)
            if not self.should_divide_transformation_by_group:
                self.data_transformed_without_group = Transformator.transform_data(self.data_transformed_without_group, self.VARIABLES, self.transform)
            else:
                self.data_transformed_without_group = self.data_transformed_with_group.copy()

    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified database."""
        file_path = ('data/clean_databases/population_results_autonomic_aging(20yGroups).csv' 
                    if self.database == 'aa' else 'data/clean_databases/population_results_bruno.csv')
        return pd.read_csv(file_path)

    def _create_bp_population(self) -> pd.DataFrame:
        """Create blood pressure population column based on SBP values."""
        conditions = [
            (self.data_transformed_without_group['mean_sbp'] <= self.mnv),
            (self.data_transformed_without_group['mean_sbp'] > self.mnv) & (self.data_transformed_without_group['mean_sbp'] < self.mhv),
            (self.data_transformed_without_group['mean_sbp'] >= self.mhv)
        ]
        values = ['normal_bp', 'intermediate_bp', 'high_bp']
        self.data_transformed_without_group['bp_population'] = np.select(conditions, values, default='unknown')
        return self.data_transformed_without_group

    def perform_anova_analysis(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform one-way ANOVA analysis for each variable across population groups.
        
        Returns:
            Dictionary containing ANOVA results for each bp_population and variable
        """
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        for bp_pop in bp_populations:
            bp_data = self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop]
            untransformed_bp_data = self.untransformed_data[self.untransformed_data['bp_population'] == bp_pop]
            if len(bp_data) < 2:
                continue
            results[bp_pop] = {}
            
            for var in self.VARIABLES:
                groups = [group[var].values for name, group in bp_data.groupby('population_group')]
                untransformed_groups = [group[var].values for name, group in untransformed_bp_data.groupby('population_group')]
                
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    group_means = [group.mean() for group in groups]
                    group_medians = [np.quantile(group, 0.5) for group in groups]
                    group_sds = [group.std() for group in groups]
                    group_names = [name for name, _ in bp_data.groupby('population_group')]
                    group_means_dict = dict(zip(group_names, group_means))
                    group_medians_dict = dict(zip(group_names, group_medians))
                    group_q1 = [np.quantile(group, 0.25) for group in groups]
                    group_q3 = [np.quantile(group, 0.75) for group in groups]
                    group_sds_dict = dict(zip(group_names, group_sds))
                    group_q1_dict = dict(zip(group_names, group_q1))
                    group_q3_dict = dict(zip(group_names, group_q3))

                    untransformed_group_means = [group.mean() for group in untransformed_groups]
                    untransformed_group_medians = [np.quantile(group, 0.5) for group in untransformed_groups]
                    untransformed_group_q1 = [np.quantile(group, 0.25) for group in untransformed_groups]
                    untransformed_group_q3 = [np.quantile(group, 0.75) for group in untransformed_groups]
                    untransformed_group_sds = [group.std() for group in untransformed_groups]
                    untransformed_group_names = [name for name, _ in untransformed_bp_data.groupby('population_group')]
                    untransformed_group_means_dict = dict(zip(untransformed_group_names, untransformed_group_means))
                    untransformed_group_sds_dict = dict(zip(untransformed_group_names, untransformed_group_sds))
                    untransformed_group_medians_dict = dict(zip(untransformed_group_names, untransformed_group_medians))
                    untransformed_group_q1_dict = dict(zip(untransformed_group_names, untransformed_group_q1))
                    untransformed_group_q3_dict = dict(zip(untransformed_group_names, untransformed_group_q3))

                    results[bp_pop][var] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'group_means': group_means_dict,
                        'untransformed_group_means': untransformed_group_means_dict,
                        'group_medians': group_medians_dict,
                        'untransformed_group_medians': untransformed_group_medians_dict,
                        'group_q1': group_q1_dict,
                        'untransformed_group_q1': untransformed_group_q1_dict,
                        'group_q3': group_q3_dict,
                        'untransformed_group_q3': untransformed_group_q3_dict,
                        'group_sds': group_sds_dict,
                        'untransformed_group_sds': untransformed_group_sds_dict
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
            bp_data = self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop]
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
        bp_data = self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop]
        
        if len(bp_data) < 2:
            return None
            
        return pairwise_tukeyhsd(
            endog=bp_data[variable],
            groups=bp_data['population_group'],
            alpha=0.05
        )

    def perform_shapiro_test(self, use_transformed = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Perform Shapiro-Wilk test for each variable across population groups."""
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        for bp_pop in bp_populations:
            bp_data = None
            if (use_transformed):
                bp_data = self.data_transformed_with_group[self.data_transformed_with_group['bp_population'] == bp_pop]
            else:
                bp_data = self.untransformed_data[self.data_transformed_with_group['bp_population'] == bp_pop]
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
    
    def save_distribution_plots(self):
        """Save distribution plots for each variable across population groups."""
        for bp_pop in ['normal_bp', 'intermediate_bp', 'high_bp']:
            for group in self.data_transformed_with_group['population_group'].unique():
                os.makedirs(f'results/normality_tests/distribution_plots', exist_ok=True)
                group_data = self.data_transformed_with_group[self.data_transformed_with_group['bp_population'] == bp_pop]
                group_data = group_data[group_data['population_group'] == group]
                if len(group_data) < 2:
                    continue
                for var in self.VARIABLES:
                    plt.figure(figsize=(10, 6))
                    if (var == 'mean_nn' and group == 'DMB'):
                        print(f'results/normality_tests/distribution_plots/{self.database}_{bp_pop}_{group}_{var}_transformed.png')
                        sns.histplot(group_data[var], bins=6)
                    else:
                        sns.histplot(group_data[var])
                    plt.title(f'{self.database}_{bp_pop}_{group}_{var} transformed, n={len(group_data)}')
                    plt.savefig(f'results/normality_tests/distribution_plots/{self.database}_{bp_pop}_{group}_{var}_transformed.png')
                    plt.clf()
        for bp_pop in ['normal_bp', 'intermediate_bp', 'high_bp']:
            for group in self.untransformed_data['population_group'].unique():
                os.makedirs(f'results/normality_tests/distribution_plots', exist_ok=True)
                group_data = self.untransformed_data[self.untransformed_data['bp_population'] == bp_pop]
                group_data = group_data[group_data['population_group'] == group]
                if len(group_data) < 2:
                    continue
                for var in self.VARIABLES:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(group_data[var])
                    plt.title(f'{self.database}_{bp_pop}_{group}_{var} untransformed, n={len(group_data)}')
                    plt.savefig(f'results/normality_tests/distribution_plots/{self.database}_{bp_pop}_{group}_{var}_untransformed.png')
                    plt.clf()

    def save_qq_plots(self):
        """Save QQ plots for each variable across population groups."""
        os.makedirs(f'results/normality_tests/qq_plots', exist_ok=True)
        for bp_pop in ['normal_bp', 'intermediate_bp', 'high_bp']:
            for group in self.data_transformed_with_group['population_group'].unique():
                group_data = self.data_transformed_with_group[self.data_transformed_with_group['bp_population'] == bp_pop]
                group_data = group_data[group_data['population_group'] == group]
                if len(group_data) < 2:
                    continue
                for var in self.VARIABLES:
                    plt.figure(figsize=(10, 6))
                    stats.probplot(group_data[var], plot=plt)
                    plt.title(f'{self.database}_{bp_pop}_{group}_{var} transformed, n={len(group_data)}')
                    plt.savefig(f'results/normality_tests/qq_plots/{self.database}_{bp_pop}_{group}_{var}_transformed.png')
                    plt.clf()
        for bp_pop in ['normal_bp', 'intermediate_bp', 'high_bp']:
            for group in self.untransformed_data['population_group'].unique():
                group_data = self.untransformed_data[self.untransformed_data['bp_population'] == bp_pop]
                group_data = group_data[group_data['population_group'] == group]
                if len(group_data) < 2:
                    continue
                for var in self.VARIABLES:
                    plt.figure(figsize=(10, 6))
                    stats.probplot(group_data[var], plot=plt)
                    plt.title(f'{self.database}_{bp_pop}_{group}_{var} untransformed, n={len(group_data)}')
                    plt.savefig(f'results/normality_tests/qq_plots/{self.database}_{bp_pop}_{group}_{var}_untransformed.png')
                    plt.clf()

    def save_results(self):
        """Save Shapiro-Wilk test results to files."""
        self.save_distribution_plots()
        self.save_qq_plots()
        shapiro_results = self.perform_shapiro_test()
        for bp_pop, variables in shapiro_results.items():
            os.makedirs('results/normality_tests', exist_ok=True)
            
            output_file = f'results/normality_tests/shapiro_wilk_{self.database}_{bp_pop}.txt'
            # remove file if it exists
            if os.path.exists(output_file):
                os.remove(output_file)
            for group, variables in shapiro_results[bp_pop].items():
                with open(output_file, 'a') as f:
                    n = len(self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop])
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
        
        shapiro_results = self.perform_shapiro_test(False)
        for bp_pop, variables in shapiro_results.items():
            os.makedirs('results/normality_tests', exist_ok=True)
            
            output_file = f'results/normality_tests/shapiro_wilk_untransformed_{self.database}_{bp_pop}.txt'
            # remove file if it exists
            if os.path.exists(output_file):
                os.remove(output_file)
            for group, variables in shapiro_results[bp_pop].items():
                with open(output_file, 'a') as f:
                    n = len(self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop])
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
            n = len(self.data_transformed_without_group[self.data_transformed_without_group['bp_population'] == bp_pop])
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
                        if stats.get('group_means') and stats.get('untransformed_group_means'):
                            f.write("\nGroup Means:\n")
                            for group, mean in stats.get('group_means').items():
                                f.write(f"{group}: {mean:.4f} ± {stats.get('group_sds')[group]:.4f} | ")
                            f.write("\nUntransformed Group Means:\n")
                            for group, mean in stats.get('untransformed_group_means').items():
                                f.write(f"{group}: {mean:.4f} ± {stats.get('untransformed_group_sds')[group]:.4f} | ")
                            f.write("\n")
                            f.write("\nGroup Medians:\n")
                            for group, median in stats.get('group_medians').items():
                                f.write(f"{group}: {median:.4f} ({stats.get('group_q1')[group]:.4f}, {stats.get('group_q3')[group]:.4f})| ")
                            f.write("\nUntransformed Group Medians:\n")
                            for group, median in stats.get('untransformed_group_medians').items():
                                f.write(f"{group}: {median:.4f} ({stats.get('untransformed_group_q1')[group]:.4f}, {stats.get('untransformed_group_q3')[group]:.4f})| ")
                    else:
                        f.write("Insufficient data for analysis\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
