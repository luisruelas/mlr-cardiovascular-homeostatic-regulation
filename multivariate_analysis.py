import argparse
import os
import shutil

from scripts.src.helpers.bivariate_analysis import BivariateAnalysis
from scripts.src.helpers.multivariate_analysis import MultivariateAnalysis
from scripts.src.helpers.univariate_analysis import UnivariateAnalysis
# how to run: pipenv run python -m scripts.src.multivariate_analysis --database aa --transform log --mnv 500 --mhv 600 --pearson_r_threshold 0.5
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    # univariate_analysis = UnivariateAnalysis(
    #     database=args.database,
    #     mnv=args.mnv,
    #     mhv=args.mhv,
    #     transform=args.transform,
    # )
    # univariate_analysis.save_results()

    # bivariate_analysis = BivariateAnalysis(
    #     database=args.database,
    #     mnv=args.mnv,
    #     mhv=args.mhv,
    #     transform=args.transform,
    #     pearson_r_threshold=args.pearson_r_threshold,
    # )

    # bivariate_analysis.create_simple_regression_plots()
    # bivariate_analysis.create_pearson_correlation_heatmaps()
    # bivariate_analysis.create_spearman_correlation_heatmaps()
    multivariate_analysis = MultivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
        transform=args.transform,
    )
    multivariate_analysis.save_results()

if __name__ == "__main__":
    main()