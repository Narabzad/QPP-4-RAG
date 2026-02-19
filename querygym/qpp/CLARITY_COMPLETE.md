# ✅ Clarity QPP Computation - COMPLETE

## Summary

**All Clarity QPP metrics have been successfully computed and integrated into your existing results!**

- ⏱️ **Computation Time**: 18.6 seconds
- 📊 **Files Processed**: 62/62 (100% success rate)
- 🔢 **Total Queries**: 3,472 queries with Clarity scores
- ✅ **Status**: COMPLETE

---

## Results Overview

### Score Statistics

| Metric | Value |
|--------|-------|
| **Mean Clarity** | 7.19 |
| **Median Clarity** | 7.05 |
| **Std Deviation** | 1.16 |
| **Range** | 4.81 - 12.69 |

### Distribution
- **Most queries** (66%) have Clarity scores between **6.0 and 8.0**
- **13% of queries** have scores below 6.0
- **21% of queries** have scores above 8.0

### Score Histogram
```
 4.0- 6.0: ██████████ (463 queries, 13%)
 6.0- 8.0: ██████████████████████████████████████████████████ (2,282 queries, 66%)
 8.0-10.0: █████████████ (633 queries, 18%)
10.0-12.0: █ (88 queries, 3%)
12.0-14.0: (6 queries, <1%)
```

---

## Files Updated

All 62 CSV files in `/future/u/negara/home/set_based_QPP/querygym/qpp/` have been updated:

### Query Expansion Methods (31 methods × 2 retrievers = 62 files)

✅ **Original queries**
- Pyserini: avg 7.19
- Cohere: avg 6.96

✅ **GenQR (5 trials)**
- Pyserini: avg 7.74
- Cohere: avg 6.93

✅ **GenQR Ensemble (5 trials)**
- Pyserini: avg 7.88
- Cohere: avg 7.02

✅ **MUGI (5 trials)**
- Pyserini: avg 7.33
- Cohere: avg 6.87

✅ **QA Expand (5 trials)**
- Pyserini: avg 7.34
- Cohere: avg 6.93

✅ **Query2Doc (5 trials)**
- Pyserini: avg 7.33
- Cohere: avg 6.85

✅ **Query2E (5 trials)**
- Pyserini: avg 7.28
- Cohere: avg 6.78

---

## Key Findings

### Retriever Comparison
- **Pyserini** retrievals generally have **higher Clarity scores** (avg 7.5) than **Cohere** retrievals (avg 6.9)
- This suggests Pyserini retrievals may be more focused/coherent

### Query Expansion Methods
- **GenQR Ensemble** achieves highest Clarity (7.88 avg with Pyserini)
- **Query2E** has lowest Clarity (6.78 avg with Cohere)
- All methods show similar patterns across retrievers

---

## Technical Details

### Challenge & Solution

**Problem**: The MS MARCO v2.1 segmented index doesn't store pre-computed document vectors (term vectors) to save space. The standard `get_document_vector()` method returns `None`.

**Solution**: Created a custom implementation that:
1. Retrieves raw document text using `doc().raw()`
2. Analyzes text with the same analyzer as queries
3. Computes term frequencies from analyzed text
4. Builds RM1 (Relevance Model 1) from term frequencies
5. Computes Clarity via KL divergence

### Parameters
- **k = 10**: Top-10 retrieved documents
- **term_num = 100**: Top-100 terms from RM1
- **mu = 1000**: Dirichlet smoothing

### Formula
```
Clarity = Σ P(w|Q) × log₂(P(w|Q) / P(w|D))
```

---

## Files Created

### Main Scripts
1. **`run_clarity_from_text.py`** - Parallel computation script (20 workers)
2. **`verify_clarity_scores.py`** - Verification and statistics script
3. **`monitor_clarity.sh`** - Progress monitoring script

### Test/Debug Scripts
4. **`test_clarity_from_text.py`** - Single-query test
5. **`test_clarity_single.py`** - Original debugging test
6. **`test_doc_vector_access.py`** - Index access debugging

### Documentation
7. **`CLARITY_RESULTS_SUMMARY.md`** - Detailed technical summary
8. **`CLARITY_COMPLETE.md`** - This file

### Log Files
9. **`clarity_from_text.log`** - Full execution log

---

## How to Use the Results

### View a specific file
```bash
cd /future/u/negara/home/set_based_QPP/querygym/qpp
head -10 post_retrieval_original_original_pyserini_qpp_metrics.csv
```

### Compare Clarity across methods
```python
import pandas as pd
import glob

files = glob.glob("post_retrieval_*_pyserini_qpp_metrics.csv")
for f in files:
    df = pd.read_csv(f)
    method = f.split('_')[2]
    avg_clarity = df['clarity-score-k10'].mean()
    print(f"{method}: {avg_clarity:.3f}")
```

### Correlation with retrieval performance
```python
# Merge with your retrieval evaluation metrics
# to analyze correlation between Clarity and performance
```

---

## Screen Session

The computation ran in a screen session named `clarity_qpp`:
```bash
# View the log
tail -f /future/u/negara/home/set_based_QPP/querygym/qpp/clarity_from_text.log

# The session has already completed and exited
```

---

## Verification

Run the verification script anytime:
```bash
cd /future/u/negara/home/set_based_QPP/querygym/qpp
python verify_clarity_scores.py
```

---

## Next Steps

You can now use these Clarity scores for:

1. **Query Performance Prediction**: Correlate Clarity with retrieval metrics (nDCG, MAP, etc.)
2. **Query Difficulty Estimation**: Lower Clarity = harder queries
3. **Method Comparison**: Compare query expansion methods by their Clarity scores
4. **Retriever Analysis**: Compare Pyserini vs Cohere retrieval quality
5. **Query Selection**: Identify queries that need improvement

---

**🎉 Clarity QPP computation successfully completed!**

*Generated: February 7, 2026*
