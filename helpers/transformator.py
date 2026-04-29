import numpy as np
import pandas as pd
from scipy import stats
from typing import List


class Transformator:
    """Class for data transformations."""
    
    @staticmethod
    def transform_data(data: pd.DataFrame, variables: List[str], transform: str) -> pd.DataFrame:
        """
        Transform data using Box-Cox or Yeo-Johnson transformation.
        """
        if transform is None:
            return data
        transformed_data = data.copy()
        if transform == 'log':
            for var in variables:
                transformed_data[var] = np.log(data[var])
        else:
            for var in variables:
                transformed_data[var] = stats.boxcox(data[var])[0]
        return transformed_data
        
    @staticmethod
    def transform_data_by_group(data: pd.DataFrame, variables: List[str], transform: str) -> pd.DataFrame:
        """
        Transform data using Box-Cox or Yeo-Johnson transformation.
        
        Args:
            data: DataFrame containing the data to transform
            variables: List of variable names to transform
            transform: Transformation method ('box' for Box-Cox, 'yeo' for Yeo-Johnson)
        
        Returns:
            DataFrame with transformed variables
        """
        if transform is None:
            return data
        
        # Create a copy to avoid modifying the original data
        transformed_data = data.copy()
        # Group by both bp_population and population_group
        for (bp_pop, pop_group), group_data in data.groupby(['bp_population', 'population_group']):
            # Get the indices for this group
            group_indices = group_data.index
            
            # Transform each variable
            for var in variables:
                # Get the variable data for this group
                var_data = group_data[var].values
                
                # Skip if there's insufficient data
                if len(var_data) < 2:
                    continue
                
                try:
                    if transform == 'log':
                        transformed_values = np.log(var_data)
                    elif transform == 'box':
                        # Box-Cox requires positive values
                        if np.any(var_data <= 0):
                            # Add a small constant to make values positive
                            var_data = var_data - np.min(var_data) + 1e-6
                        transformed_values, _ = stats.boxcox(var_data)
                    elif transform == 'yeo':
                        # Yeo-Johnson can handle negative values
                        transformed_values, _ = stats.yeojohnson(var_data)
                    else:
                        # If transform is neither 'box' nor 'yeo', skip transformation
                        continue
                    
                    # Update the transformed data at the correct indices
                    transformed_data.loc[group_indices, var] = transformed_values
                    
                except Exception as e:
                    # If transformation fails, keep original values
                    print(f"Warning: Transformation failed for {var} in group {bp_pop}_{pop_group}: {e}")
                    continue
        return transformed_data
