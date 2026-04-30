# Univariate, Bivariate, and Multivariate analysis for basic Cardiovascular Variables

This project performs a statistical analysis of cardiovascular time series data to study how heart rate variability and blood pressure relate to each other across different populations. The analysis covers two independent databases and uses univariate, bivariate, and multivariate statistical methods to characterize group differences and detect significant associations.

---

## What are we studying?

We analyze four physiological variables measured from beat-to-beat cardiovascular recordings:

| Variable | Meaning |
|----------|---------|
| `mean_nn` | Mean inter-beat interval (IBI) in milliseconds — inversely related to heart rate |
| `sd_nn` | Standard deviation of IBI — a measure of heart rate variability (HRV) |
| `mean_sbp` | Mean systolic blood pressure (SBP) in mmHg |
| `sd_sbp` | Standard deviation of systolic blood pressure — beat-to-beat BP variability |

The goal is to understand how these variables differ across age groups or clinical groups, and whether they are statistically related to each other.

---

## Databases

### `aa` — Autonomic Ageing (PhysioNet)

A publicly available dataset from [PhysioNet](https://physionet.org/) that includes healthy subjects across a wide age range. Subjects are divided into three age groups:

- `18-29y` — Young adults
- `30-49y` — Middle-aged adults
- `50+y` — Older adults

This database allows us to study how autonomic cardiovascular regulation changes with normal ageing.

### `bruno` — Dr. Bruno Estañol's Clinical Database

A clinical database collected at our institution, containing three groups of subjects:

- `Control` — Healthy control subjects with no known metabolic disease
- `DMA` — Patients with Type 2 Diabetes (T2D) of **recent diagnosis** (shorter disease duration)
- `DMB` — Patients with Type 2 Diabetes (T2D) with **non-recent diagnosis** (longer disease duration)

This database lets us study how diabetes and its progression affect cardiovascular autonomic function.

---

## Setup: installing dependencies

> If you have never used Python or a terminal before, follow these steps carefully. This only needs to be done once.

### Step 1 — Install Python 3.12

Download and install Python 3.12 from [https://www.python.org/downloads/](https://www.python.org/downloads/). During installation, make sure the option **"Add Python to PATH"** is checked.

### Step 2 — Install Pipenv

Open a terminal (on Mac: press `Cmd + Space`, type `Terminal`, press Enter) and run:

```bash
pip install pipenv
```

Pipenv is a tool that creates an isolated Python environment for this project and installs all required packages automatically. This prevents conflicts with other Python projects on your computer.

### Step 3 — Navigate to this folder

In the terminal, go to the `scripts/src` folder. Replace the path below with wherever you saved this project:

```bash
cd /path/to/DrBruno/scripts/src
```

### Step 4 — Install all packages

```bash
pipenv install
```

This reads the `Pipfile` and installs all required libraries:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Loading and handling tabular data (CSV files) |
| `matplotlib` | Generating all plots and figures |
| `seaborn` | Statistical visualizations (heatmaps, distributions) |
| `statsmodels` | Statistical tests and regression models |
| `scikit-posthocs` | Post-hoc tests after ANOVA (e.g., Tukey HSD) |
| `starbars` | Significance bars on plots |
| `visibility-graph` | Graph-based signal analysis |

You are now ready to run the analysis.

---

## How to run

All analyses are launched from the root project folder (one level above `scripts/`) using the following command format:

```bash
pipenv run python -m multivariate_analysis --database aa --transform log --mnv 500 --mhv 600 --pearson_r_threshold 0.5
```

### Arguments explained

| Argument | Required | Description |
|----------|----------|-------------|
| `--database` | Yes | Which dataset to analyze: `aa` (Autonomic Ageing) or `bruno` (Dr. Estañol's database) |
| `--mnv` | Yes | **Maximum Normotensive Value** — the upper SBP threshold (in mmHg) for the normal blood pressure group. Subjects with SBP ≤ this value are classified as `normal_bp`. |
| `--mhv` | Yes | **Minimum Hypertensive Value** — the lower SBP threshold (in mmHg) for the high blood pressure group. Subjects with SBP ≥ this value are classified as `high_bp`. Subjects between the two thresholds form the `intermediate_bp` group. |
| `--transform` | No | Transformation applied to the data before parametric tests. Options: `log` (natural log), `box` (Box-Cox), `yeo` (Yeo-Johnson). Omit this argument to run without transformation. |
| `--pearson_r_threshold` | No | Minimum absolute Pearson correlation coefficient to highlight in heatmaps. Cells with `r ≥ threshold` **and** `p < 0.05` are visually emphasized. Default is `0.5`. |

### Example commands

Run on the autonomic ageing database with log transformation:
```bash
pipenv run python -m multivariate_analysis --database aa --transform log --mnv 500 --mhv 600 --pearson_r_threshold 0.5
```

Run on Bruno's database without any transformation:
```bash
pipenv run python -m multivariate_analysis --database bruno --mnv 500 --mhv 600
```

> **Note:** The analysis currently focuses on the `normal_bp` group. The `mnv` and `mhv` values define that group's boundary — subjects outside that range are excluded from the main analysis.

---

## Output folders

All results are saved inside `scripts/src/results/`. The folder is organized as follows:

---

### `normality_tests/`

Before applying parametric statistical tests, we assess whether each variable follows a normal (Gaussian) distribution within each group.

**Contents:**

- `shapiro_wilk_*.txt` — Results of the **Shapiro-Wilk normality test** for each variable, group, and database. Reports the test statistic, p-value, skewness, and kurtosis. A p-value below 0.05 indicates the variable is **not normally distributed**.
- `distribution_plots/` — Histograms showing the frequency distribution of each variable per group, both before and after transformation.
- `qq_plots/` — Q-Q (quantile-quantile) plots. If the data points follow the diagonal line, the variable is approximately normal.

Also at the root of `results/`:
- `anova_aa_normal_bp.txt` / `anova_bruno_normal_bp.txt` — Results of the **ANOVA** (parametric) and **Kruskal-Wallis** (non-parametric) group comparison tests, followed by **Tukey's HSD post-hoc** pairwise comparisons. Includes group means, medians, and sample sizes.

---

### `correlation_heatmaps/`

Examines pairwise linear associations between the four variables within each group.

**Contents:**

- `.png` files — Color-coded correlation matrices. Positive correlations are shown in red, negative in blue, and no correlation in white. Cells where the correlation is statistically significant (p < 0.05) **and** the effect is strong (|r| ≥ threshold) are bold and bordered.
- `.txt` files — The numeric version of each heatmap: the full correlation matrix with r-values and p-values.

Two methods are computed:
- **Pearson** — Assumes linearity and normality; measures linear correlation.
- **Spearman** — Rank-based, non-parametric; more robust to outliers and non-normal distributions.

File naming format: `{database}_{bp_group}_{population_group}_{method}_correlation.{ext}`

Example: `bruno_normal_bp_Control_pearson_correlation.png`

---

### `simple_regression_plots/`

Visualizes the pairwise relationship between each pair of variables using scatter plots with a fitted regression line (OLS — Ordinary Least Squares).

**Contents:**

- `.png` files — One plot per variable pair per group. Each plot shows data points scattered and a straight regression line through them. The steeper and tighter the line, the stronger the linear relationship.

File naming format: `{database}_{group}_{variable1}_vs_{variable2}_simple_regression.png`

Example: `bruno_Control_mean_nn_vs_sd_nn_simple_regression.png`

---

### `multivariate_analysis/`

Builds a multiple linear regression model for each variable, using the other three as predictors simultaneously. This allows us to assess the unique contribution of each predictor while controlling for the others.

**Contents:**

- `.txt` files — Full OLS regression summaries for each dependent variable, group, and database. Each file includes:
  - R-squared (how much of the variance is explained by the model)
  - F-statistic and p-value (whether the model as a whole is significant)
  - Individual coefficient estimates, standard errors, t-statistics, and p-values for each predictor
  - **VIF (Variance Inflation Factor)** — checks for multicollinearity (whether predictors are too correlated with each other, which can make individual estimates unreliable)

- `coeficients_graphs/` — Line plots showing the regression coefficients for each predictor across groups. Significant coefficients (p < 0.05, |β| ≥ threshold) are marked with asterisks. The background turns colored when the overall model is significant and R² is above a meaningful threshold.

File naming format: `multivariate_{database}_{bp_group}_{population_group}.txt`

Example: `multivariate_bruno_normal_bp_Control.txt`

---

## Troubleshooting

**`command not found: pipenv`** — Pipenv is not installed or not on your PATH. Re-run `pip install pipenv` and restart your terminal.

**`ModuleNotFoundError`** — You are running `python` directly instead of `pipenv run python`. Always prefix commands with `pipenv run`.

**`No such file or directory`** — Make sure you are running the command from the root project folder (the one containing the `scripts/` directory), not from inside `scripts/src/`.
