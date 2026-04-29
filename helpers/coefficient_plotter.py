import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class CoefficientPlotter:
    """
    A class for creating coefficient plots for regression analysis results.
    """
    
    def __init__(self, 
                 variable_name_mapping: Dict[str, Dict[str, str]], 
                 line_style_for_variables: Dict[str, Dict[str, str]],
                 significance_level: float = 0.05,
                 r_squared_threshold: float = 0.3,
                 beta_threshold: float = 0.4,
                 light_yellow: Tuple[float, float, float, float] = (0.890, 0.969, 0.475, 0.8)):
        """
        Initialize the CoefficientPlotter.
        
        Args:
            variable_name_mapping: Dictionary mapping variable names to full names and abbreviations
            line_style_for_variables: Dictionary mapping variables to their line styles and colors
            significance_level: P-value threshold for significance (default: 0.05)
            light_yellow: RGBA tuple for background color when model is significant
        """
        self.variable_name_mapping = variable_name_mapping
        self.line_style_for_variables = line_style_for_variables
        self.significance_level = significance_level
        self.r_squared_threshold = r_squared_threshold
        self.beta_threshold = beta_threshold
        self.light_yellow = light_yellow
        self.white = (1, 1, 1, 1)
    
    def plot_coefficients(self,
                         coefficients_dict: Dict[str, float],
                         coefficients_p_values: Dict[str, float],
                         model_p_value: float,
                         model_r_squared: float,
                         population_group: str,
                         condition: str,
                         dependent_variable: str,
                         x_range: List[float] = [-3, 3],
                         y_range: List[float] = [-3, 3],
                         extra_info: bool = False,
                         save_path: Optional[str] = None,
                         show_plot: bool = False) -> Dict[str, Dict[str, List[float]]]:
        """
        Create a coefficient plot for regression results.
        
        Args:
            coefficients_dict: Dictionary of coefficient values (excluding 'const')
            coefficients_p_values: Dictionary of p-values for each coefficient
            model_p_value: Overall model F-statistic p-value
            model_r_squared: Overall model R-squared value
            population_group: Name of the population group
            condition: Name of the condition
            dependent_variable: Name of the dependent variable
            x_range: Range for x-axis (default: [-3, 3])
            y_range: Range for y-axis (default: [-3, 3])
            extra_info: Whether to include coefficient values in labels
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary containing regression coordinates for each variable
        """
        # Clear any existing plot
        plt.clf()
        
        # Create axes and set background color if model is significant
        ax = plt.axes()
        is_model_significant_and_high = model_p_value <= self.significance_level and model_r_squared > self.r_squared_threshold
        print('****dependent_variable****', dependent_variable)
        print('****population_group****', population_group)
        print('****condition****', condition)
        print('****r_squared****', model_r_squared)
        print('****significance_level****', self.significance_level)
        print('****is_model_significant_and_high****', is_model_significant_and_high)
        if is_model_significant_and_high:
            ax.set_facecolor(self.light_yellow)
        else:
            ax.set_facecolor(self.white)
        
        # Dictionary to store regression coordinates
        regression_coordinates_dict = {}
        
        # Plot coefficient lines
        for variable_name, coefficient_value in coefficients_dict.items():
            if variable_name == 'const':
                continue
                
            # Calculate y values for the line
            x = np.array(x_range)
            y = coefficient_value * x
            
            # Get p-value and determine significance
            pvalue = coefficients_p_values.get(variable_name, 1.0)
            is_coefficient_significant_and_high = is_model_significant_and_high and pvalue < self.significance_level and abs(coefficient_value) >= self.beta_threshold
            pvalue_asterisk = '*' if is_coefficient_significant_and_high else ''
            linewidth = 4 if is_coefficient_significant_and_high else 1
            
            # Create label
            coefficient_rounded = round(coefficient_value, 4)
            variable_abv = self.variable_name_mapping.get(variable_name, {}).get('abv', variable_name)
            label = f'{variable_abv}{pvalue_asterisk}'
            if extra_info:
                label += f' (Coef: {coefficient_rounded})'
            
            # Store coordinates
            key = f"{population_group}{condition}{dependent_variable}{variable_name}"
            regression_coordinates_dict[key] = {'x': x.tolist(), 'y': y.tolist()}
            
            # Get line style
            line_style = self.line_style_for_variables.get(variable_name, {})
            color = line_style.get('color', 'black')
            linestyle = line_style.get('linestyle', '-')
            
            # Plot the line
            plt.plot(x, y, 
                    label=label, 
                    color=color, 
                    linestyle=linestyle, 
                    linewidth=linewidth)
        
        # Customize plot appearance
        self._customize_plot_appearance(ax, dependent_variable, coefficients_dict, 
                                       model_p_value, y_range)
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        # Clean up
        if not show_plot:
            plt.close()
        
        return regression_coordinates_dict
    
    def _customize_plot_appearance(self, 
                                  ax, 
                                  dependent_variable: str, 
                                  coefficients_dict: Dict[str, float],
                                  model_p_value: float,
                                  y_range: List[float]):
        """
        Apply consistent styling to the plot.
        
        Args:
            ax: Matplotlib axes object
            dependent_variable: Name of the dependent variable
            coefficients_dict: Dictionary of coefficient values
            model_p_value: Overall model F-statistic p-value
            y_range: Range for y-axis
        """
        # Add grid and axis lines
        plt.grid(color='gray', linestyle='--', linewidth=0.3, dashes=(10, 10))
        plt.axhline(0, color='gray', linewidth=0.35, linestyle='-')
        plt.axvline(0, color='gray', linewidth=0.35, linestyle='-')
        
        # Customize axis spines
        for spine in ['bottom', 'top', 'right', 'left']:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(0.3)
        
        # Set axis limits
        plt.ylim(y_range)
        
        # Set legend and labels
        plt.legend(fontsize=15)
        
        # Set y-label
        dependent_var_abv = self.variable_name_mapping.get(dependent_variable, {}).get('abv', dependent_variable)
        plt.ylabel(dependent_var_abv, fontsize=15)
        
        # Create x-label from independent variables
        independent_vars = [var for var in coefficients_dict.keys() if var != 'const']
        independent_var_abvs = [
            self.variable_name_mapping.get(var, {}).get('abv', var) 
            for var in independent_vars
        ]
        plt.xlabel(f'({", ".join(independent_var_abvs)})')
    
    def plot_multiple_coefficients(self,
                                  regression_results: List[Dict],
                                  output_directory: str,
                                  x_range: List[float] = [-3, 3],
                                  y_range: List[float] = [-3, 3],
                                  extra_info: bool = False) -> Dict[str, Dict[str, List[float]]]:
        """
        Create coefficient plots for multiple regression results.
        
        Args:
            regression_results: List of dictionaries containing regression results.
                               Each dict should have keys: 'coefficients_dict', 'coefficients_p_values',
                               'model_p_value', 'population_group', 'condition', 'dependent_variable'
            output_directory: Directory to save the plots
            x_range: Range for x-axis (default: [-3, 3])
            y_range: Range for y-axis (default: [-3, 3])
            extra_info: Whether to include coefficient values in labels
            
        Returns:
            Dictionary containing all regression coordinates
        """
        all_regression_coordinates = {}
        
        for result in regression_results:
            # Extract required information
            coefficients_dict = result['coefficients_dict']
            coefficients_p_values = result['coefficients_p_values']
            model_p_value = result['model_p_value']
            model_r_squared = result['model_r_squared']
            population_group = result['population_group']
            condition = result['condition']
            dependent_variable = result['dependent_variable']
            
            # Generate filename
            filename = f'coefficients_{population_group}_{condition}_{dependent_variable}.png'
            save_path = os.path.join(output_directory, filename)
            
            # Create plot
            coords = self.plot_coefficients(
                coefficients_dict=coefficients_dict,
                coefficients_p_values=coefficients_p_values,
                model_p_value=model_p_value,
                model_r_squared=model_r_squared,
                population_group=population_group,
                condition=condition,
                dependent_variable=dependent_variable,
                x_range=x_range,
                y_range=y_range,
                extra_info=extra_info,
                save_path=save_path
            )
            
            # Merge coordinates
            all_regression_coordinates.update(coords)
        
        return all_regression_coordinates


def create_coefficient_plot(coefficients_dict: Dict[str, float],
                           coefficients_p_values: Dict[str, float],
                           model_p_value: float,
                           model_r_squared: float,
                           population_group: str,
                           condition: str,
                           dependent_variable: str,
                           variable_name_mapping: Dict[str, Dict[str, str]],
                           line_style_for_variables: Dict[str, Dict[str, str]],
                           significance_level: float = 0.05,
                           x_range: List[float] = [-3, 3],
                           y_range: List[float] = [-3, 3],
                           extra_info: bool = False,
                           save_path: Optional[str] = None,
                           show_plot: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """
    Convenience function to create a coefficient plot.
    
    This is a wrapper around the CoefficientPlotter class for simple use cases.
    
    Args:
        coefficients_dict: Dictionary of coefficient values (excluding 'const')
        coefficients_p_values: Dictionary of p-values for each coefficient
        model_p_value: Overall model F-statistic p-value
        model_r_squared: Overall model R-squared value
        population_group: Name of the population group
        condition: Name of the condition
        dependent_variable: Name of the dependent variable
        variable_name_mapping: Dictionary mapping variable names to full names and abbreviations
        line_style_for_variables: Dictionary mapping variables to their line styles and colors
        significance_level: P-value threshold for significance (default: 0.05)
        x_range: Range for x-axis (default: [-3, 3])
        y_range: Range for y-axis (default: [-3, 3])
        extra_info: Whether to include coefficient values in labels
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        Dictionary containing regression coordinates for each variable
    """
    plotter = CoefficientPlotter(
        variable_name_mapping=variable_name_mapping,
        line_style_for_variables=line_style_for_variables,
        significance_level=significance_level
    )
    
    return plotter.plot_coefficients(
        coefficients_dict=coefficients_dict,
        coefficients_p_values=coefficients_p_values,
        model_p_value=model_p_value,
        model_r_squared=model_r_squared,
        population_group=population_group,
        condition=condition,
        dependent_variable=dependent_variable,
        x_range=x_range,
        y_range=y_range,
        extra_info=extra_info,
        save_path=save_path,
        show_plot=show_plot
    )
