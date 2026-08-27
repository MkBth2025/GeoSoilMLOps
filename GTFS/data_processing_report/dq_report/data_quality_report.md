# Data Quality and Statistical Analysis

- **Working directory:** `G:\Researching\Geo_MLops_2026`
- **Output directory:** `G:\Researching\Geo_MLops_2026\data_processing_report\dq_report`

- **Rows checked:** 332
- **Columns analyzed:** AC, Distance, Depth, Ucs_class, PI/FF, PI, FF, LL, PL
- **Duplicate rows:** 0
- **Overall status:** **PASS**

## Missingness

| Column | Missing % |
|---|---:|
| AC | 0.00% |
| Distance | 0.00% |
| Depth | 0.00% |
| Ucs_class | 0.00% |
| PI/FF | 0.00% |
| PI | 0.00% |
| FF | 0.00% |
| LL | 0.00% |
| PL | 0.00% |

## Constraint Checks

| Rule | Condition | Passed | Violations | Example indexes |
|---|---|:---:|---:|---|
| AC_nonnegative | AC must be >= 0 | Yes | 0 | - |
| Distance_nonnegative | Distance must be >= 0 | Yes | 0 | - |
| Depth_nonnegative | Depth must be >= 0 | Yes | 0 | - |
| Ucs_class_nonnegative | Ucs_class must be >= 0 | Yes | 0 | - |
| PI/FF_nonnegative | PI/FF must be >= 0 | Yes | 0 | - |
| PI_nonnegative | PI must be >= 0 | Yes | 0 | - |
| FF_nonnegative | FF must be >= 0 | Yes | 0 | - |
| LL_nonnegative | LL must be >= 0 | Yes | 0 | - |
| PL_nonnegative | PL must be >= 0 | Yes | 0 | - |

## Descriptive Statistics

- CSV: `table2_stats.csv`

## Correlations

- Pearson CSV: `corr_pearson.csv`
- Spearman CSV: `corr_spearman.csv`
- Heatmaps: `heatmap_corr_pearson.png`, `heatmap_corr_spearman.png`

**Pearson correlations with target `AC`:**

| Feature | r |
|---|---:|
| PI/FF | 0.612 |
| FF | -0.549 |
| PI | 0.380 |
| LL | 0.262 |
| Distance | -0.234 |
| Depth | -0.221 |
| Ucs_class | 0.210 |
| PL | -0.055 |

## Normality Analysis

- Summary CSV: `normality_summary.csv`
- Combined publication figure: `all normality.png`
- Interpretation: for the Shapiro-Wilk test, p >= alpha does not reject normality; p < alpha indicates evidence against normality.

| Column | n | Shapiro W | p-value | Decision | Plot |
|---|---:|---:|---:|---|---|
| AC | 332 | 0.82399 | 9.66648e-19 | Non-normal | `AC_normality.png` |
| Distance | 332 | 0.87371 | 7.42897e-16 | Non-normal | `Distance_normality.png` |
| Depth | 332 | 0.81555 | 3.60525e-19 | Non-normal | `Depth_normality.png` |
| Ucs_class | 332 | 0.62530 | 2.72693e-26 | Non-normal | `Ucs_class_normality.png` |
| PI/FF | 332 | 0.87025 | 4.42327e-16 | Non-normal | `PI_FF_normality.png` |
| PI | 332 | 0.91734 | 1.49444e-12 | Non-normal | `PI_normality.png` |
| FF | 332 | 0.82374 | 9.38813e-19 | Non-normal | `FF_normality.png` |
| LL | 332 | 0.95208 | 6.26047e-09 | Non-normal | `LL_normality.png` |
| PL | 332 | 0.98258 | 0.000473297 | Non-normal | `PL_normality.png` |