# QPP Correlation Analysis

## Overview

This analysis computes correlations between QPP metrics and actual performance metrics across all 31 query variations (original + 6 methods × 5 trials) for each of the 56 queries.

**Methodology**:
- For each QPP metric, correlations are calculated per-query across all 31 variations
- Correlations are then averaged across all 56 queries
- Both **Pearson correlation** (linear) and **Kendall Tau** (rank-based) are reported

**Performance Metrics Analyzed**:
- Nugget scores: `strict_vital_score`, `strict_all_score`, `vital_score`, `all_score`
- Retrieval metrics: `ndcg@10`, `recall@100`

**QPP Metrics Analyzed**:
- Pre-retrieval metrics (analyzed with both Pyserini and Cohere retrieval)
- Post-retrieval Pyserini metrics (analyzed with Pyserini retrieval)
- Post-retrieval Cohere metrics (analyzed with Cohere retrieval)

---

## Key Findings

### Top Correlations with Nugget All Score

#### Pearson Correlation (Linear Relationship)

| Rank | QPP Metric | Type | Retrieval | Pearson r | Kendall τ |
|------|------------|------|-----------|-----------|-----------|
| 1 | ql | Pre | Pyserini | **0.2444** | 0.1955 |
| 2 | SCQ-sum | Pre | Pyserini | **0.2441** | 0.1946 |
| 3 | IDF-sum | Pre | Pyserini | **0.2440** | 0.1945 |
| 4 | SCQ-max | Pre | Pyserini | **0.2436** | 0.1705 |
| 5 | IDF-max | Pre | Pyserini | **0.2434** | 0.1756 |
| 6 | sigma-max | Post | Pyserini | **0.2339** | 0.1859 |
| 7 | sigma-x0.5 | Post | Pyserini | **0.2335** | 0.1765 |
| 8 | nqc-no-norm-k100 | Post | Pyserini | **0.2330** | 0.1899 |
| 9 | RSD | Post | Pyserini | **0.2301** | 0.1855 |
| 10 | smv-no-norm-k100 | Post | Pyserini | **0.2301** | 0.1855 |

**Insight**: Pre-retrieval metrics (query length, IDF, SCQ) show the strongest correlations with nugget scores for Pyserini retrieval.

#### Kendall Tau (Rank Correlation)

| Rank | QPP Metric | Type | Retrieval | Pearson r | Kendall τ |
|------|------------|------|-----------|-----------|-----------|
| 1 | ql | Pre | Pyserini | 0.2444 | **0.1955** |
| 2 | SCQ-sum | Pre | Pyserini | 0.2441 | **0.1946** |
| 3 | IDF-sum | Pre | Pyserini | 0.2440 | **0.1945** |
| 4 | nqc-no-norm-k100 | Post | Pyserini | 0.2330 | **0.1899** |
| 5 | sigma-max | Post | Pyserini | 0.2339 | **0.1859** |

**Insight**: Rank-based correlations are consistently lower than linear correlations, suggesting some non-linear relationships.

### Top Correlations with Retrieval nDCG@10

#### Pearson Correlation

| Rank | QPP Metric | Type | Retrieval | Pearson r | Kendall τ |
|------|------------|------|-----------|-----------|-----------|
| 1 | nqc-no-norm-k100 | Post | Cohere | **0.3286** | 0.2251 |
| 2 | RSD | Post | Cohere | **0.3196** | 0.2176 |
| 3 | smv-no-norm-k100 | Post | Cohere | **0.3196** | 0.2176 |
| 4 | nqc-norm-k100 | Post | Cohere | **0.2905** | 0.1952 |
| 5 | sigma-max | Post | Cohere | **0.2902** | 0.2071 |
| 6 | smv-norm-k100 | Post | Cohere | **0.2897** | 0.1992 |
| 7 | nqc-norm-k100 | Post | Pyserini | **0.2182** | 0.1537 |
| 8 | smv-norm-k100 | Post | Pyserini | **0.1993** | 0.1432 |
| 9 | bert_qpp_cross_encoder | Post | Pyserini | **0.1472** | 0.0917 |
| 10 | IDF-max | Pre | Pyserini | **0.1002** | 0.0742 |

**Insight**: Post-retrieval metrics (especially NQC, RSD, SMV) show much stronger correlations with nDCG@10 than pre-retrieval metrics. Cohere post-retrieval metrics perform better than Pyserini.

---

## QSDQPP Correlation Results

### QSDQPP with Pyserini Retrieval

| Metric | Pearson r | Kendall τ |
|--------|-----------|-----------|
| **Nugget All Score** | 0.0702 | 0.0360 |
| **Nugget Strict All Score** | 0.0877 | 0.0529 |
| **Retrieval nDCG@10** | 0.0105 | 0.0073 |
| **Retrieval Recall@100** | 0.0265 | 0.0246 |

**Interpretation**: Weak positive correlations with Pyserini retrieval. QSDQPP shows modest predictive power for nugget scores but minimal correlation with retrieval metrics.

### QSDQPP with Cohere Retrieval

| Metric | Pearson r | Kendall τ |
|--------|-----------|-----------|
| **Nugget All Score** | -0.0162 | -0.0038 |
| **Nugget Strict All Score** | -0.0163 | -0.0023 |
| **Retrieval nDCG@10** | -0.1064 | -0.0904 |
| **Retrieval Recall@100** | -0.0658 | -0.0415 |

**Interpretation**: Very weak to negative correlations with Cohere retrieval. QSDQPP shows poor predictive power for Cohere retrieval performance.

**Key Finding**: Despite QSDQPP ranking 7th in oracle analysis for Cohere, its correlation with actual performance is weak. This suggests QSDQPP may be selecting good queries for different reasons than direct linear correlation.

---

## Comparison: Pre vs Post-Retrieval Metrics

### Nugget All Score Correlations

**Pre-Retrieval Metrics (Pyserini)**:
- Best: ql (0.2444), SCQ-sum (0.2441), IDF-sum (0.2440)
- Average: ~0.20-0.24

**Post-Retrieval Metrics (Pyserini)**:
- Best: sigma-max (0.2339), sigma-x0.5 (0.2335), nqc-no-norm-k100 (0.2330)
- Average: ~0.20-0.23

**Conclusion**: Pre-retrieval and post-retrieval metrics show similar correlation strengths with nugget scores for Pyserini.

### Retrieval nDCG@10 Correlations

**Pre-Retrieval Metrics**:
- Best: IDF-max (0.1002) with Pyserini
- Average: ~0.00-0.10 (weak)

**Post-Retrieval Metrics**:
- Best: nqc-no-norm-k100 (0.3286) with Cohere
- Average: ~0.20-0.33 (moderate to strong)

**Conclusion**: Post-retrieval metrics show **much stronger** correlations with retrieval nDCG@10, especially for Cohere retrieval.

---

## Comparison: Pyserini vs Cohere Retrieval

### Pre-Retrieval Metrics

**Nugget All Score**:
- Pyserini: 0.20-0.24 (stronger)
- Cohere: 0.10-0.16 (weaker)

**Retrieval nDCG@10**:
- Pyserini: -0.15 to 0.10 (weak, sometimes negative)
- Cohere: -0.15 to 0.01 (weak, sometimes negative)

**Conclusion**: Pre-retrieval metrics correlate better with Pyserini nugget scores but show weak correlations with retrieval metrics for both methods.

### Post-Retrieval Metrics

**Nugget All Score**:
- Pyserini: 0.20-0.23 (moderate)
- Cohere: -0.06 to 0.00 (very weak, sometimes negative)

**Retrieval nDCG@10**:
- Pyserini: 0.15-0.22 (moderate)
- Cohere: 0.29-0.33 (strong)

**Conclusion**: Post-retrieval metrics show strong correlations with Cohere nDCG@10 but weak correlations with Cohere nugget scores.

---

## Methodology Details

### Correlation Calculation

For each QPP metric and performance metric pair:

1. **Per-Query Analysis**: For each of the 56 queries:
   - Collect QPP values for all 31 variations
   - Collect corresponding performance metric values
   - Calculate Pearson and Kendall Tau correlations

2. **Aggregation**: Average correlations across all 56 queries

3. **Filtering**: Only queries with at least 3 valid data points are included

### QPP Metric Types

- **Pre-retrieval**: Analyzed separately with Pyserini and Cohere retrieval metrics
  - Examples: ql, IDF-*, SCQ-*, SCS-*, avgICTF, qsdqpp_predicted_ndcg

- **Post-retrieval Pyserini**: Analyzed with Pyserini retrieval metrics
  - Examples: clarity-score-k10, wig-norm-k100, nqc-norm-k100, etc.

- **Post-retrieval Cohere**: Analyzed with Cohere retrieval metrics
  - Examples: clarity-score-k10, wig-norm-k100, nqc-norm-k100, etc.

### Performance Metrics

**Nugget Scores** (from `nugget_scores.retrieval` or `nugget_scores.retrieval_cohere`):
- `strict_vital_score`: Strict matching, vital nuggets only
- `strict_all_score`: Strict matching, all nuggets
- `vital_score`: Relaxed matching, vital nuggets only
- `all_score`: Relaxed matching, all nuggets

**Retrieval Metrics** (from `retrieval_metrics.pyserini` or `retrieval_metrics.cohere`):
- `ndcg@10`: Normalized Discounted Cumulative Gain at rank 10
- `recall@100`: Recall at rank 100

---

## Statistical Interpretation

### Correlation Strength Guidelines

| |r| Range | Interpretation |
|---|--------|----------------|
| 0.00 - 0.10 | Negligible |
| 0.10 - 0.30 | Weak |
| 0.30 - 0.50 | Moderate |
| 0.50 - 0.70 | Strong |
| 0.70 - 1.00 | Very Strong |

### Key Observations

1. **Strongest Correlations**: Pre-retrieval metrics with nugget scores (r ≈ 0.24)
2. **Moderate Correlations**: Post-retrieval metrics with nDCG@10 for Cohere (r ≈ 0.29-0.33)
3. **Weak Correlations**: Most QPP metrics with retrieval metrics for Pyserini (r < 0.10)
4. **Negative Correlations**: Some pre-retrieval metrics show negative correlations with Cohere retrieval metrics

---

## Files Generated

### Full Results
- `qpp_correlations_full.csv`: Complete correlation results with all metrics and standard deviations

### Simplified Results
- `qpp_correlations_simple.csv`: Key correlation metrics only (Pearson and Kendall means)

### Columns in Simplified File
- `qpp_metric`: Name of QPP metric
- `qpp_type`: pre_pyserini, pre_cohere, post_pyserini, or post_cohere
- `retrieval_method`: pyserini or cohere
- `nugget_all_score_pearson_mean`: Average Pearson correlation with nugget all score
- `nugget_all_score_kendall_mean`: Average Kendall Tau with nugget all score
- `nugget_strict_all_score_pearson_mean`: Average Pearson correlation with strict nugget all score
- `nugget_strict_all_score_kendall_mean`: Average Kendall Tau with strict nugget all score
- `retrieval_ndcg@10_pearson_mean`: Average Pearson correlation with nDCG@10
- `retrieval_ndcg@10_kendall_mean`: Average Kendall Tau with nDCG@10
- `retrieval_recall@100_pearson_mean`: Average Pearson correlation with Recall@100
- `retrieval_recall@100_kendall_mean`: Average Kendall Tau with Recall@100

---

## Recommendations

### For Query Selection

1. **Nugget Score Prediction (Pyserini)**:
   - Use pre-retrieval metrics: ql, SCQ-sum, IDF-sum (r ≈ 0.24)
   - Post-retrieval metrics also effective: sigma-max, nqc-no-norm-k100 (r ≈ 0.23)

2. **nDCG@10 Prediction (Cohere)**:
   - Use post-retrieval metrics: nqc-no-norm-k100, RSD, smv-no-norm-k100 (r ≈ 0.29-0.33)
   - Pre-retrieval metrics are weak (r < 0.10)

3. **QSDQPP**:
   - Shows weak correlations but good oracle performance
   - May be capturing non-linear relationships
   - Consider using in ensemble with other metrics

### For Research

1. **Non-Linear Relationships**: Weak linear correlations but good oracle performance suggest non-linear relationships worth exploring

2. **Retrieval Method Differences**: Significant differences between Pyserini and Cohere correlations suggest method-specific QPP strategies

3. **Metric Selection**: Different QPP metrics are optimal for different performance targets (nugget scores vs retrieval metrics)

---

**Date**: February 9, 2026  
**Total QPP Metrics Analyzed**: 43 (including QSDQPP with both retrieval methods)  
**Total Queries**: 56  
**Variations per Query**: 31 (original + 6 methods × 5 trials)
