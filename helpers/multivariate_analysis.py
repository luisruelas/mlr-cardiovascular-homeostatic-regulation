import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm

from scripts.src.helpers.coefficient_plotter import CoefficientPlotter
from .transformator import Transformator

class MultivariateAnalysis:
    # Reuse the same variables and mapping from UnivariateAnalysis and BivariateAnalysis
    VARIABLES = ['mean_nn', 'sd_nn', 'mean_sbp', 'sd_sbp']
    VARIABLE_MAPPING = {
        'mean_nn': {'full_name': 'Mean IBI', 'abv': 'meanIBI'},
        'sd_nn': {'full_name': 'SD IBI', 'abv': 'sdIBI'},
        'mean_sbp': {'full_name': 'Mean SBP', 'abv': 'meanSBP'},
        'sd_sbp': {'full_name': 'SD SBP', 'abv': 'sdSBP'},
    }

    groups_translation = {
        'DMA': 'T2DA',
        'DMB': 'T2DB',
    }
    
    LINE_STYLE_FOR_VARIABLES = {
        'mean_nn': {'color': 'red', 'linestyle': '-'},
        'sd_nn': {'color': 'red', 'linestyle': '--'},
        'mean_sbp': {'color': 'blue', 'linestyle': '-'},
        'sd_sbp': {'color': 'blue', 'linestyle': '--'},
    }

    def __init__(self, database: str, mnv: float, mhv: float, transform: str = None):
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
        self.data = self._create_bp_population()
        if transform is not None:
            self.data = Transformator.transform_data_by_group(self.data, self.VARIABLES, transform)
        self.data = self._normalize_data()

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

    def _perform_multiple_regression(self, y: np.ndarray, X: np.ndarray, independent_vars: List[str]) -> Tuple[str, object]:
        """
        Perform multiple linear regression using statsmodels and return summary and results object.
        
        Args:
            y: Dependent variable
            X: Independent variables matrix
            independent_vars: List of independent variable names
            
        Returns:
            Tuple containing (summary_string, results_object) or (None, None) if insufficient data
        """
        if len(y) < len(X[0]) + 2:  # Need at least n_features + 2 observations
            return None, None
        
        # Convert X to DataFrame with column names
        X_df = pd.DataFrame(X, columns=independent_vars)
        X_with_const = sm.add_constant(X_df)
        
        # Fit the OLS model
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        # Return both summary string and results object
        return str(results.summary()), results

    def perform_multivariate_analysis(self) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Perform multiple regression analysis for each variable as dependent variable
        with all others as independent variables, for each bp_population and population_group.
        
        Returns:
            Nested dictionary with results organized by bp_population, population_group, and dependent variable
        """
        results = {}
        bp_populations = ['normal_bp', 'intermediate_bp', 'high_bp']
        # print all existing bp_populations with count
        print(self.data['bp_population'].unique(), self.data['bp_population'].value_counts())
        # print all existing population_groups with count
        print(self.data['population_group'].unique(), self.data['population_group'].value_counts())
        # exit()
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
                    regression_summary, regression_results = self._perform_multiple_regression(y, X, independent_vars)
                    
                    # Store results with variable names
                    results[bp_pop][pop_group][dependent_var] = {
                        'dependent_variable': dependent_var,
                        'independent_variables': independent_vars,
                        'summary': regression_summary,
                        'results_object': regression_results
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
                    f.write("Multiple Regression Analysis Results\n")
                    f.write(f"Database: {self.database}\n")
                    f.write(f"Blood Pressure Population: {bp_pop.replace('_', ' ').title()}\n")
                    f.write(f"Population Group: {pop_group}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for dependent_var, analysis in dependent_vars.items():
                        dependent_name = self.VARIABLE_MAPPING[dependent_var]['abv']
                        independent_names = [self.VARIABLE_MAPPING[var]['abv'] for var in analysis['independent_variables']]

                        # Example: Get parameters from actual regression results instead of hardcoding
                        params = self.get_regression_parameters_for_plot(
                            bp_population=bp_pop,  # or 'intermediate_bp', 'high_bp'
                            population_group=pop_group,  # Use actual population group from your data
                            dependent_variable=dependent_var
                        )
                        pop_group_for_path = self.groups_translation.get(pop_group, pop_group)
                        if params is not None:
                            CoefficientPlotter(variable_name_mapping=self.VARIABLE_MAPPING, 
                                             line_style_for_variables=self.LINE_STYLE_FOR_VARIABLES).plot_coefficients(
                                coefficients_dict=params['coefficients_dict'],
                                coefficients_p_values=params['coefficients_p_values'],
                                model_p_value=params['model_p_value'],
                                model_r_squared=params['model_r_squared'],
                                population_group=params['population_group'],
                                condition=params['condition'],
                                dependent_variable=params['dependent_variable'],
                                x_range=[-2, 2],
                                y_range=[-2.5, 2.5],
                                extra_info=False,
                                save_path=f'results/multivariate_analysis/coeficients_graphs/{self.database}/coef_graph_{self.database}_{dependent_var}_{bp_pop}_{pop_group_for_path}.png'
                            )
                        else:
                            print("No regression results found for the specified parameters")
                        
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
    
    def create_coefficient_plots(self, output_directory: str = 'results/multivariate_analysis/coefficient_plots', 
                                significance_level: float = 0.05, extra_info: bool = False):
        """
        Create coefficient plots for all regression results.
        
        Args:
            output_directory: Directory to save the plots
            significance_level: P-value threshold for significance (default: 0.05)
            extra_info: Whether to include coefficient values in labels
        """
        # Get regression results
        results = self.perform_multivariate_analysis()
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Initialize coefficient plotter
        plotter = CoefficientPlotter(
            variable_name_mapping=self.VARIABLE_MAPPING,
            line_style_for_variables=self.LINE_STYLE_FOR_VARIABLES,
            significance_level=significance_level
        )
        
        all_regression_coordinates = {}
        for bp_pop, pop_groups in results.items():
            for pop_group, dependent_vars in pop_groups.items():
                for dependent_var, analysis in dependent_vars.items():
                    # Skip if no results object (insufficient data)
                    if analysis['results_object'] is None:
                        continue
                    
                    results_obj = analysis['results_object']
                    
                    # Extract coefficients and p-values (excluding constant)
                    coefficients_dict = results_obj.params.drop('const').to_dict()
                    coefficients_p_values = results_obj.pvalues.drop('const').to_dict()
                    model_p_value = results_obj.f_pvalue
                    model_r_squared = results_obj.rsquared
                    # Create filename
                    output_directory = f'results/multivariate_analysis/coefficient_plots/{self.database}'
                    filename = f'coefficients_{bp_pop}_{pop_group}_{dependent_var}.png'
                    save_path = os.path.join(output_directory, filename)
                    
                    # Create plot
                    coords = plotter.plot_coefficients(
                        coefficients_dict=coefficients_dict,
                        coefficients_p_values=coefficients_p_values,
                        model_p_value=model_p_value,
                        model_r_squared=model_r_squared,
                        population_group=pop_group,
                        condition=bp_pop,
                        dependent_variable=dependent_var,
                        extra_info=extra_info,
                        save_path=save_path
                    )
                    
                    # Store coordinates
                    all_regression_coordinates.update(coords)
                    
                    print(f"Created coefficient plot: {filename}")
        
        return all_regression_coordinates
    
    def get_regression_parameters_for_plot(self, bp_population: str, population_group: str, dependent_variable: str):
        """
        Extract regression parameters for a specific analysis to use in coefficient plots.
        
        Args:
            bp_population: Blood pressure population ('normal_bp', 'intermediate_bp', 'high_bp')
            population_group: Population group (e.g., '18-29y', 'Control', etc.)
            dependent_variable: Dependent variable name
            
        Returns:
            Dictionary with coefficients_dict, coefficients_p_values, and model_p_value
        """
        results = self.perform_multivariate_analysis()
        
        try:
            analysis = results[bp_population][population_group][dependent_variable]
            results_obj = analysis['results_object']
            
            if results_obj is None:
                return None
            
            # Extract parameters
            coefficients_dict = results_obj.params.drop('const').to_dict()
            coefficients_p_values = results_obj.pvalues.drop('const').to_dict()
            model_p_value = results_obj.f_pvalue
            model_r_squared = results_obj.rsquared           
            return {
                'coefficients_dict': coefficients_dict,
                'coefficients_p_values': coefficients_p_values,
                'model_p_value': model_p_value,
                'model_r_squared': model_r_squared,
                'population_group': population_group,
                'condition': bp_population,
                'dependent_variable': dependent_variable
            }
        except KeyError as e:
            print(f"Analysis not found for {bp_population}, {population_group}, {dependent_variable}: {e}")
            return None 