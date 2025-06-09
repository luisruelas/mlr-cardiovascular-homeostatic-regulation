import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm

class MultivariateAnalysis:
    # Reuse the same variables and mapping from UnivariateAnalysis and BivariateAnalysis
    VARIABLES = ['mean_nn', 'sd_nn', 'mean_sbp', 'sd_sbp']
    VARIABLE_MAPPING = {
        'mean_nn': {'full_name': 'Mean NN', 'abv': 'meanNN'},
        'sd_nn': {'full_name': 'SD NN', 'abv': 'sdNN'},
        'mean_sbp': {'full_name': 'Mean SBP', 'abv': 'meanSBP'},
        'sd_sbp': {'full_name': 'SD SBP', 'abv': 'sdSBP'},
    }

    def __init__(self, database: str, mnv: float, mhv: float):
        """
        Initialize MultivariateAnalysis class.
        
        Args:
            database: Database to analyze ('aa' or 'bruno')
            mnv: Maximum normotensive value
            mhv: Minimum hypertensive value
        """
        self.database = database
        self.mnv = mnv
        self.mhv = mhv
        self.data = self._load_data()
        self.data = self._normalize_data()
        self.data = self._create_bp_population()

    def _normalize_data(self) -> pd.DataFrame:
        """Normalize the data. Only variable columns"""
        self.data[self.VARIABLES] = self.data[self.VARIABLES].apply(lambda x: (x - x.mean()) / x.std())
        return self.data

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

    def _perform_multiple_regression(self, y: np.ndarray, X: np.ndarray) -> str:
        """
        Perform multiple linear regression using statsmodels and return summary.
        
        Args:
            y: Dependent variable
            X: Independent variables matrix
            
        Returns:
            String containing OLS regression summary or None if insufficient data
        """
        if len(y) < len(X[0]) + 2:  # Need at least n_features + 2 observations
            return None
        
        # Add constant term for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit the OLS model
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        # Return the summary as string
        return str(results.summary())

    def perform_multivariate_analysis(self) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Perform multiple regression analysis for each variable as dependent variable
        with all others as independent variables, for each bp_population and population_group.
        
        Returns:
            Nested dictionary with results organized by bp_population, population_group, and dependent variable
        """
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        
        for bp_pop in bp_populations:
            bp_data = self.data[self.data['bp_population'] == bp_pop]
            if len(bp_data) < 2:
                continue
                
            results[bp_pop] = {}
            population_groups = bp_data['population_group'].unique()
            
            for pop_group in population_groups:
                group_data = bp_data[bp_data['population_group'] == pop_group]
                
                if len(group_data) < len(self.VARIABLES):  # Need at least as many observations as variables
                    continue
                    
                results[bp_pop][pop_group] = {}
                
                # For each variable as dependent variable
                for dependent_var in self.VARIABLES:
                    independent_vars = [var for var in self.VARIABLES if var != dependent_var]
                    
                    # Prepare data
                    y = group_data[dependent_var].values
                    X = group_data[independent_vars].values
                    
                    # Perform regression
                    regression_summary = self._perform_multiple_regression(y, X)
                    
                    # Store results with variable names
                    results[bp_pop][pop_group][dependent_var] = {
                        'dependent_variable': dependent_var,
                        'independent_variables': independent_vars,
                        'summary': regression_summary
                    }
        
        return results

    def save_results(self):
        """Save multivariate analysis results to files."""
        results = self.perform_multivariate_analysis()
        
        # Create results directory if it doesn't exist
        os.makedirs('results/multivariate_analysis', exist_ok=True)
        
        # Clear existing files
        if os.path.exists('results/multivariate_analysis'):
            for file in os.listdir('results/multivariate_analysis'):
                if file.startswith(f'multivariate_{self.database}'):
                    os.remove(os.path.join('results/multivariate_analysis', file))
        
        for bp_pop, pop_groups in results.items():
            for pop_group, dependent_vars in pop_groups.items():
                output_file = f'results/multivariate_analysis/multivariate_{self.database}_{bp_pop}_{pop_group}.txt'
                
                with open(output_file, 'w') as f:
                    f.write(f"Multiple Regression Analysis Results\n")
                    f.write(f"Database: {self.database}\n")
                    f.write(f"Blood Pressure Population: {bp_pop.replace('_', ' ').title()}\n")
                    f.write(f"Population Group: {pop_group}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for dependent_var, analysis in dependent_vars.items():
                        dependent_name = self.VARIABLE_MAPPING[dependent_var]['abv']
                        independent_names = [self.VARIABLE_MAPPING[var]['abv'] for var in analysis['independent_variables']]
                        
                        f.write(f"{dependent_name}({','.join(independent_names)})\n")
                        f.write("-" * 50 + "\n")
                        
                        if analysis['summary'] is not None:
                            f.write(analysis['summary'])
                        else:
                            f.write("Insufficient data for analysis\n")
                        
                        f.write("\n" + "=" * 80 + "\n\n")

    def get_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table with regression equations in the format requested.
        
        Returns:
            DataFrame with results organized by bp_population, population_group, and regression equations
        """
        results = self.perform_multivariate_analysis()
        summary_data = []
        
        for bp_pop, pop_groups in results.items():
            for pop_group, dependent_vars in pop_groups.items():
                # Add bp_population and population_group as separate rows
                summary_data.append({
                    'bp_population': bp_pop.replace('_', ''),
                    'population_group': pop_group,
                    'regression_equation': ''
                })
                
                for dependent_var, analysis in dependent_vars.items():
                    dependent_name = self.VARIABLE_MAPPING[dependent_var]['abv']
                    independent_names = [self.VARIABLE_MAPPING[var]['abv'] for var in analysis['independent_variables']]
                    
                    equation = f"{dependent_name}({','.join(independent_names)})"
                    
                    summary_data.append({
                        'bp_population': '',
                        'population_group': '',
                        'regression_equation': equation
                    })
        
        return pd.DataFrame(summary_data)

    def save_summary_table(self):
        """Save the summary table to a CSV file."""
        summary_df = self.get_summary_table()
        output_file = f'results/multivariate_analysis/multivariate_summary_{self.database}.csv'
        summary_df.to_csv(output_file, index=False)
        
        # Also save as formatted text
        output_txt = f'results/multivariate_analysis/multivariate_summary_{self.database}.txt'
        with open(output_txt, 'w') as f:
            f.write(f"Multivariate Analysis Summary - Database: {self.database}\n")
            f.write("=" * 80 + "\n\n")
            
            current_bp_pop = None
            current_pop_group = None
            
            for _, row in summary_df.iterrows():
                if row['bp_population']:  # New bp_population
                    current_bp_pop = row['bp_population']
                    f.write(f"\n{current_bp_pop}\n")
                
                if row['population_group']:  # New population_group
                    current_pop_group = row['population_group']
                    f.write(f"{current_pop_group}\n")
                
                if row['regression_equation']:  # Regression equation
                    f.write(f"{row['regression_equation']}\n") 