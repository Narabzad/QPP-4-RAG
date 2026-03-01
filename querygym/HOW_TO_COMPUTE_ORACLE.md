# How to Compute Oracle QPP Performance

This guide explains how to compute oracle QPP performance, which measures the best possible performance achievable by selecting query variants based on QPP predictions.

## What is Oracle Performance?

Oracle performance answers the question: "If we could perfectly predict which query variant would perform best using QPP, what would be the best performance we could achieve?"

For each QPP metric:
1. For each query, select the variant with the **highest QPP score**
2. Measure the actual performance (nugget scores, nDCG, etc.) of that selected variant
3. Aggregate across all queries to get overall oracle performance

## Prerequisites

1. **Consolidated query data**: All query variants with QPP predictions and performance metrics
   - File: `querygym/consolidated_query_data.json`
   - This file contains all reformulations with their QPP metrics and nugget scores

2. **QPP predictions**: Pre-retrieval and post-retrieval QPP metrics
   - Location: `querygym/qpp/`
   - Pre-retrieval: `pre_retrieval_*.csv` files
   - Post-retrieval: `post_retrieval_*.csv` files

3. **Nuggetizer scores**: Evaluation results for all query variants
   - Location: `querygym/rag_nuggetized_eval/`

## Step-by-Step Guide

### Step 1: Ensure Consolidated Data is Up-to-Date

The consolidated data file should contain:
- All query variants (31 methods × 56 queries)
- QPP metrics (pre-retrieval and post-retrieval)
- Nuggetizer scores
- Retrieval metrics (nDCG, MAP, etc.)

If you need to regenerate it:

```bash
cd querygym
python consolidate_query_data.py
```

### Step 2: Run Oracle Analysis

Use the `analyze_qpp_oracle_performance.py` script:

```bash
cd querygym
python analyze_qpp_oracle_performance.py \
    --consolidated-file consolidated_query_data.json \
    --output-dir qpp_oracle_analysis \
    --retrieval-method pyserini \
    --performance-metric nugget_all_score
```

**Parameters**:
- `--consolidated-file`: Path to consolidated query data JSON file
- `--output-dir`: Directory to save oracle analysis results
- `--retrieval-method`: `pyserini` or `cohere`
- `--performance-metric`: Which metric to optimize for
  - `nugget_all_score` (recommended)
  - `nugget_vital_score`
  - `nugget_strict_all_score`
  - `nugget_strict_vital_score`
  - `ndcg@5`, `ndcg@10`, `map`, etc.

### Step 3: Run for Both Retrieval Methods

```bash
# For Pyserini
python analyze_qpp_oracle_performance.py \
    --consolidated-file consolidated_query_data.json \
    --output-dir qpp_oracle_analysis \
    --retrieval-method pyserini \
    --performance-metric nugget_all_score

# For Cohere
python analyze_qpp_oracle_performance.py \
    --consolidated-file consolidated_query_data.json \
    --output-dir qpp_oracle_analysis \
    --retrieval-method cohere \
    --performance-metric nugget_all_score
```

### Step 4: View Results

The script generates CSV files in the output directory:

- `qpp_oracle_performance_pyserini.csv` - Oracle results for Pyserini
- `qpp_oracle_performance_cohere.csv` - Oracle results for Cohere
- `qpp_oracle_performance_combined.csv` - Combined results

Each CSV file contains:
- `qpp_metric`: Name of the QPP metric
- `oracle_score`: Best achievable performance using this QPP metric
- `mean_score`: Mean performance across all variants
- `improvement`: Improvement over mean (oracle - mean)
- `improvement_pct`: Percentage improvement

## Understanding Oracle Results

### Example Output

```
qpp_metric,oracle_score,mean_score,improvement,improvement_pct
pre_SCQ-avg,0.5107,0.3634,0.1473,40.5%
post_nqc-no-norm-k100,0.5332,0.3634,0.1698,46.7%
```

**Interpretation**:
- Using `pre_SCQ-avg` to select variants achieves 0.5107 nugget_all_score
- This is 40.5% better than the mean performance (0.3634)
- Using `post_nqc-no-norm-k100` achieves even better: 0.5332 (46.7% improvement)

### Oracle vs. Actual Performance

Oracle performance represents the **upper bound** of what QPP-based selection can achieve. In practice:
- Oracle assumes perfect QPP-based selection
- Actual performance will be lower due to:
  - QPP prediction errors
  - Correlation between QPP and performance
  - Other factors affecting selection

## Advanced Usage

### Analyze Multiple Performance Metrics

```bash
# Analyze multiple metrics at once
for metric in nugget_all_score nugget_vital_score ndcg@10; do
    python analyze_qpp_oracle_performance.py \
        --consolidated-file consolidated_query_data.json \
        --output-dir qpp_oracle_analysis \
        --retrieval-method pyserini \
        --performance-metric $metric
done
```

### Filter QPP Metrics

The script automatically analyzes all available QPP metrics. To focus on specific metrics, you can modify the script or filter results after generation.

### Generate Visualizations

The oracle analysis script can generate plots showing:
- Oracle performance by QPP metric
- Comparison between retrieval methods
- Scatter plots of QPP vs. performance

Check the `qpp_oracle_analysis/` directory for generated plots.

## Key Findings from Oracle Analysis

Based on the results in `qpp_oracle_analysis/`:

1. **Best Pre-Retrieval QPP**: `pre_SCQ-avg` achieves strong oracle performance
2. **Best Post-Retrieval QPP**: `post_nqc-no-norm-k100` achieves the highest oracle performance
3. **Retrieval Method Matters**: Cohere retrieval shows better oracle performance than Pyserini
4. **Utility Gap**: Oracle performance shows significant improvement potential over mean performance

## Related Files

- `querygym/analyze_qpp_oracle_performance.py` - Main oracle analysis script
- `querygym/consolidate_query_data.py` - Script to create consolidated data
- `querygym/qpp_oracle_analysis/` - Directory with oracle analysis results
- `querygym/qpp_oracle_analysis/final/` - Filtered and final oracle results

## Citation

Oracle analysis methodology is described in the paper. Oracle performance represents the theoretical upper bound for QPP-based query variant selection.
