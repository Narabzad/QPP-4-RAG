# Correlation Analysis: Nugget Scores vs QPP Metrics

## Overview
This analysis examines the correlation between nugget evaluation scores and Query Performance Prediction (QPP) metrics for both Pyserini and Cohere retrieval methods.

## Data Summary
- **Total queries**: 56
- **Total reformulations**: 1,736 samples per retrieval method
- **Nugget metrics analyzed**: 
  - `strict_vital_score`
  - `strict_all_score`
  - `vital_score`
  - `all_score`
- **QPP metrics analyzed**:
  - Clarity scores (k10, k100)
  - WIG (Weighted Information Gain) - normalized and non-normalized (k100, k1000)
  - NQC (Normalized Query Clarity) - normalized and non-normalized (k100)
  - SMV (Score Magnitude Variance) - normalized and non-normalized (k100)
  - Sigma metrics (x0.5, max)
  - RSD (Robust Standard Deviation)

## Key Findings

### Pyserini Retrieval

**Top Correlations (by absolute Pearson r):**

1. **strict_all_score ↔ sigma-x0.5**: r = 0.207 (p < 0.001)
2. **all_score ↔ sigma-x0.5**: r = 0.193 (p < 0.001)
3. **strict_all_score ↔ nqc-no-norm-k100**: r = 0.164 (p < 0.001)
4. **strict_all_score ↔ sigma-max**: r = 0.162 (p < 0.001)
5. **all_score ↔ nqc-no-norm-k100**: r = 0.161 (p < 0.001)

**Observations:**
- Weak to moderate positive correlations (0.10 - 0.21)
- Strongest correlations with **sigma-x0.5** (variance-based metric)
- **Non-normalized QPP metrics** (nqc-no-norm-k100, smv-no-norm-k100) show stronger correlations than normalized versions
- **All nugget scores** show similar correlation patterns, with `strict_all_score` and `all_score` showing slightly stronger correlations

### Cohere Retrieval

**Top Correlations (by absolute Pearson r):**

1. **strict_all_score ↔ smv-no-norm-k100**: r = 0.317 (p < 0.001)
2. **strict_all_score ↔ RSD**: r = 0.317 (p < 0.001)
3. **all_score ↔ smv-no-norm-k100**: r = 0.313 (p < 0.001)
4. **all_score ↔ RSD**: r = 0.313 (p < 0.001)
5. **strict_all_score ↔ nqc-no-norm-k100**: r = 0.311 (p < 0.001)

**Observations:**
- Moderate positive correlations (0.29 - 0.32)
- **Stronger correlations than Pyserini** (approximately 1.5x stronger)
- Strongest correlations with **variance-based metrics** (SMV, RSD, NQC)
- **Non-normalized metrics** consistently outperform normalized versions
- `strict_all_score` and `all_score` show the strongest correlations

## Comparison: Pyserini vs Cohere

| Metric | Pyserini (r) | Cohere (r) | Difference |
|--------|--------------|------------|------------|
| Best correlation | 0.207 | 0.317 | +0.110 |
| SMV-based metrics | 0.135-0.155 | 0.295-0.317 | ~2x stronger |
| NQC-based metrics | 0.133-0.164 | 0.286-0.311 | ~2x stronger |
| Sigma metrics | 0.193-0.207 | 0.116-0.294 | Mixed |

**Key Differences:**
1. **Cohere shows consistently stronger correlations** across all QPP metrics
2. **Variance-based metrics** (SMV, RSD, NQC) are better predictors for Cohere retrieval
3. **Non-normalized metrics** are consistently better predictors than normalized versions for both methods

## Statistical Significance

All reported correlations are **highly statistically significant** (p < 0.001) with:
- Sample size: n = 1,736
- Both Pearson (linear) and Spearman (rank) correlations show similar patterns
- Spearman correlations are often slightly stronger, suggesting some non-linear relationships

## Implications

1. **QPP metrics are better predictors for Cohere retrieval** than Pyserini retrieval
2. **Variance-based QPP metrics** (SMV, RSD, NQC) are the most informative
3. **Non-normalized QPP metrics** should be preferred over normalized versions for nugget score prediction
4. **All nugget scores** show similar correlation patterns, suggesting they capture related aspects of retrieval quality

## Files Generated

- `pyserini_correlations.csv`: Full correlation matrix for Pyserini
- `cohere_correlations.csv`: Full correlation matrix for Cohere
- `pyserini_correlation_heatmap.png`: Visual heatmap of correlations (Pyserini)
- `cohere_correlation_heatmap.png`: Visual heatmap of correlations (Cohere)
- `pyserini_scatter_plots.png`: Scatter plots of top correlations (Pyserini)
- `cohere_scatter_plots.png`: Scatter plots of top correlations (Cohere)

## Methodology

- **Correlation methods**: Pearson (linear) and Spearman (rank-order)
- **Sample size**: 1,736 reformulations (56 queries × 31 reformulations)
- **Significance threshold**: p < 0.001
- **Visualization**: Heatmaps and scatter plots for top correlations
