import argparse

from scripts.src.helpers.bivariate_analysis import BivariateAnalysis
from scripts.src.helpers.multivariate_analysis import MultivariateAnalysis
from scripts.src.helpers.univariate_analysis import UnivariateAnalysis
# how to run: pipenv run python -m scripts.src.multivariate_analysis --database aa --mnv 500 --mhv 600
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multivariate analysis for cardiovascular data')
    parser.add_argument('--database', type=str, choices=['aa', 'bruno'], required=True,
                      help='Database to analyze (aa or bruno)')
    parser.add_argument('--transform', type=str, choices=['box', 'yeo'], default=None,
                      help='Data transformation method')
    parser.add_argument('--mnv', type=float, required=True,
                      help='Maximum normotensive value')
    parser.add_argument('--mhv', type=float, required=True,
                      help='Minimum hypertensive value')
    return parser.parse_args()

def main():
    args = parse_arguments()
    univariate_analysis = UnivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
    )
    univariate_analysis.save_results()

    bivariate_analysis = BivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
    )
    bivariate_analysis.create_pearson_correlation_heatmaps()
    bivariate_analysis.create_spearman_correlation_heatmaps()
    multivariate_analysis = MultivariateAnalysis(
        database=args.database,
        mnv=args.mnv,
        mhv=args.mhv,
    )
    multivariate_analysis.save_results()

if __name__ == "__main__":
    main()