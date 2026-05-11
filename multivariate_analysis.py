import argparse
import os
import shutil

from helpers.bivariate_analysis import BivariateAnalysis
from helpers.multivariate_analysis import MultivariateAnalysis
from helpers.univariate_analysis import UnivariateAnalysis
from helpers.exclusion_helper import ExclusionHelper

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multivariate analysis for cardiovascular data')
    parser.add_argument('--database', type=str, choices=['aa', 'bruno'], required=True,
                      help='Database to analyze (aa or bruno)')
    parser.add_argument('--transform', type=str, choices=['box', 'yeo', 'log'], default=None,
                      help='Data transformation method')
    parser.add_argument('--mnv', type=float, required=True,
                      help='Maximum normotensive value')
    parser.add_argument('--mhv', type=float, required=True,
                      help='Minimum hypertensive value')
    parser.add_argument('--pearson_r_threshold', type=float, default=0.5,
                      help='Pearson correlation r threshold')
    parser.add_argument('--age_group', type=int, choices=[10, 20], default=20,
                      help='Age grouping for AA database: 20 (default) or 10 years')
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.database == 'aa':
        aa_file = (
            f'clean_databases/population_results_autonomic_aging({args.age_group}yGroups).csv'
        )
        exclusion_helper = ExclusionHelper(
            results_path=aa_file,
            subject_info_path='clean_databases/subject_info_aa.csv',
            age_group=args.age_group,
        )
        exclusion_helper.save_exclusion_barchart(output_dir='results')

    univariate_analysis = UnivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
        transform=args.transform,
        age_group=args.age_group,
    )
    univariate_analysis.save_results()

    bivariate_analysis = BivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
        transform=args.transform,
        pearson_r_threshold=args.pearson_r_threshold,
        age_group=args.age_group,
    )
    bivariate_analysis.create_simple_regression_plots()
    bivariate_analysis.create_pearson_correlation_heatmaps()
    bivariate_analysis.create_spearman_correlation_heatmaps()

    multivariate_analysis = MultivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
        transform=args.transform,
        age_group=args.age_group,
    )
    multivariate_analysis.save_results()

if __name__ == "__main__":
    main()
