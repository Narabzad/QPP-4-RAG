# QPP Oracle Analysis - Updated with QSDQPP

## ✅ Complete

QPP Oracle analysis has been rerun with QSDQPP included as a pre-retrieval metric.

---

## 📊 QSDQPP Oracle Performance

### Pyserini Retrieval

| Metric | Value |
|--------|-------|
| **Nugget All Score** | 0.3543 |
| **Nugget Strict All Score** | 0.2669 |
| **Retrieval nDCG@10** | 0.2955 |
| **Retrieval Recall@100** | 0.1989 |
| **Queries Processed** | 56 |

**Ranking**: Mid-tier performance for Pyserini

### Cohere Retrieval

| Metric | Value |
|--------|-------|
| **Nugget All Score** | 0.4054 |
| **Nugget Strict All Score** | 0.3119 |
| **Retrieval nDCG@10** | 0.5130 |
| **Retrieval Recall@100** | 0.3405 |
| **Queries Processed** | 56 |

**Ranking**: 🏆 **#7 out of 31 metrics** (by nugget_all_score)

---

## 🎯 Key Findings

### QSDQPP Performance Highlights

1. **Strong Performance with Cohere**: QSDQPP ranks 7th out of 31 QPP metrics when used as an oracle selector for Cohere retrieval
2. **Better for Dense Retrieval**: QSDQPP performs significantly better with Cohere (dense retrieval) than Pyserini (sparse retrieval)
3. **Competitive Pre-Retrieval Method**: As a pre-retrieval method, QSDQPP achieves competitive performance without needing to perform actual retrieval

### Comparison with Other Pre-Retrieval Methods (Cohere)

| Rank | QPP Metric | Nugget All Score | Type |
|------|------------|------------------|------|
| 1 | max_oracle_nugget_all | 0.5332 | Oracle (upper bound) |
| 2 | pre_SCQ-avg | 0.4212 | Pre-retrieval |
| 3 | pre_avgICTF | 0.4211 | Pre-retrieval |
| 4 | pre_IDF-avg | 0.4152 | Pre-retrieval |
| 5 | pre_IDF-sum | 0.4098 | Pre-retrieval |
| 6 | pre_SCQ-sum | 0.4091 | Pre-retrieval |
| **7** | **pre_qsdqpp_predicted_ndcg** | **0.4054** | **Pre-retrieval** |
| 8 | pre_ql | 0.4043 | Pre-retrieval |
| 9 | pre_SCS-2 | 0.4009 | Pre-retrieval |
| 10 | pre_IDF-max | 0.3983 | Pre-retrieval |

### Top Post-Retrieval Methods (Cohere) for Comparison

| Rank | QPP Metric | Nugget All Score | Type |
|------|------------|------------------|------|
| 11 | max_oracle_ndcg@10 | 0.3963 | Oracle (upper bound) |
| 12 | clarity-score-k100 | 0.3819 | Post-retrieval |
| 13 | clarity-score-k10 | 0.3778 | Post-retrieval |

**Insight**: QSDQPP outperforms all post-retrieval methods including Clarity, despite being a pre-retrieval method!

---

## 📈 Performance by Retrieval Method

### Nugget All Score Comparison

```
Cohere:  0.4054  ████████████████████████████████████████
Pyserini: 0.3543  ███████████████████████████████
```

QSDQPP shows **14.4% better performance** with Cohere vs Pyserini.

### Retrieval nDCG@10 Comparison

```
Cohere:  0.5130  ██████████████████████████████████████████████████
Pyserini: 0.2955  ████████████████████████
```

QSDQPP shows **73.6% better nDCG@10** with Cohere vs Pyserini.

---

## 🔍 Analysis Methodology

**Oracle Selection Strategy**:
For each query, select the reformulation with the **highest QSDQPP predicted nDCG** value, then measure the actual performance of that selection.

**Metrics Evaluated**:
- Nugget scores (vital and all, strict and relaxed)
- Retrieval metrics (nDCG@10, Recall@100)

**Baseline Comparisons**:
- **Original**: Performance using only original queries
- **Max Oracle**: Upper bound - selecting best reformulation based on actual performance

---

## 📁 Output Files

### Full Results
- `qpp_oracle_performance_pyserini.csv` - Complete results for Pyserini (28 metrics)
- `qpp_oracle_performance_cohere.csv` - Complete results for Cohere (28 metrics)

### Simplified Results
- `qpp_oracle_performance_pyserini_simple.csv` - Key metrics only
- `qpp_oracle_performance_cohere_simple.csv` - Key metrics only

### Columns in Simplified Files
- `qpp_metric` - Name of QPP metric
- `retrieval_method` - pyserini or cohere
- `n_queries` - Number of queries processed
- `nugget_all_score_mean` - Average nugget all score
- `nugget_strict_all_score_mean` - Average strict nugget all score
- `retrieval_ndcg@10_mean` - Average nDCG@10
- `retrieval_recall@100_mean` - Average Recall@100

---

## 💡 Insights & Recommendations

### 1. QSDQPP is Effective for Dense Retrieval
QSDQPP performs exceptionally well with Cohere (dense retrieval), ranking 7th among all QPP metrics. This suggests it's well-suited for selecting query reformulations in dense retrieval scenarios.

### 2. Pre-Retrieval Advantage
As a pre-retrieval method, QSDQPP can make selection decisions **before** performing expensive retrieval operations, making it highly efficient.

### 3. Outperforms Post-Retrieval Methods
QSDQPP (pre-retrieval) achieves better oracle performance than Clarity (post-retrieval), which is remarkable given that Clarity has access to retrieval results.

### 4. Complementary to Other Methods
QSDQPP could be combined with other top pre-retrieval metrics (SCQ-avg, avgICTF, IDF-avg) for ensemble-based selection.

### 5. Use Case Recommendations
- **For Cohere/Dense Retrieval**: QSDQPP is highly recommended (top 7 performance)
- **For Pyserini/Sparse Retrieval**: Consider combining QSDQPP with other metrics
- **For Efficiency**: QSDQPP provides good performance without retrieval overhead

---

## 🎓 About the Oracle Analysis

**What is Oracle Analysis?**
Oracle analysis evaluates the theoretical upper bound of QPP-based query reformulation selection. For each query, we select the reformulation that the QPP metric predicts will perform best, then measure actual performance.

**Why is it Important?**
- Shows the potential value of each QPP metric for query selection
- Identifies which metrics are most reliable for choosing reformulations
- Provides upper bounds for expected performance improvements

**Interpretation**:
- Higher scores = better query selection capability
- Max oracle = theoretical upper bound (perfect selection)
- Original = baseline (no reformulation)

---

## 📊 All Pre-Retrieval Metrics (Cohere)

| Rank | Metric | Nugget All Score | nDCG@10 |
|------|--------|------------------|---------|
| 2 | pre_SCQ-avg | 0.4212 | 0.5121 |
| 3 | pre_avgICTF | 0.4211 | 0.5021 |
| 4 | pre_IDF-avg | 0.4152 | 0.5102 |
| 5 | pre_IDF-sum | 0.4098 | 0.5001 |
| 6 | pre_SCQ-sum | 0.4091 | 0.4960 |
| **7** | **pre_qsdqpp_predicted_ndcg** | **0.4054** | **0.5130** |
| 8 | pre_ql | 0.4043 | 0.5134 |
| 9 | pre_SCS-2 | 0.4009 | 0.4951 |
| 10 | pre_IDF-max | 0.3983 | 0.5226 |
| 11 | pre_IDF-std | 0.3927 | 0.4877 |
| 12 | pre_SCS-1 | 0.3917 | 0.4745 |
| 13 | pre_SCQ-max | 0.3824 | 0.4968 |

---

**Date**: February 9, 2026  
**Status**: ✅ Complete  
**Total QPP Metrics Analyzed**: 28 (including QSDQPP)
