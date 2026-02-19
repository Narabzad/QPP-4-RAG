# Filtered Oracle Performance Results

## 📁 File

**`qpp_oracle_performance_filtered.csv`**

- **96 rows** (48 pyserini + 48 cohere)
- **8 columns** (key metrics only)
- Combines both retrieval methods in one file

---

## 📊 Columns

1. **`qpp_metric`** - QPP method or oracle type
2. **`retrieval_method`** - pyserini or cohere
3. **`nugget_strict_vital_score_mean`** - Retrieval-based strict vital score
4. **`nugget_all_score_mean`** - Retrieval-based all score
5. **`genonly_strict_vital_score_mean`** - Generation-only strict vital score
6. **`genonly_all_score_mean`** - Generation-only all score
7. **`retrieval_ndcg@5_mean`** - NDCG@5 retrieval metric
8. **`retrieval_recall@100_mean`** - Recall@100 retrieval metric

---

## 📋 Row Types

### QPP Methods (31 × 2 = 62 rows)
Pre-retrieval and post-retrieval QPP methods for both pyserini and cohere:
- `pre_IDF-avg`, `pre_IDF-max`, `pre_IDF-sum`, `pre_IDF-std`
- `pre_SCQ-avg`, `pre_SCQ-max`, `pre_SCQ-sum`
- `pre_avgICTF`, `pre_ql`, `pre_SCS-1`, `pre_SCS-2`
- `pre_rv_bert`, `pre_rv_e5`, `pre_qsdqpp_predicted_ndcg`
- `bert_qpp_bi_encoder`, `bert_qpp_cross_encoder`
- `clarity-score-k10`, `clarity-score-k100`
- `nqc-norm-k100`, `nqc-no-norm-k100`
- `smv-norm-k100`, `smv-no-norm-k100`
- `sigma-x0.5`, `sigma-max`, `RSD`
- `wig-norm-k100`, `wig-no-norm-k100`, etc.

### Max Oracle - Retrieval-Based (6 × 2 = 12 rows)
Select best reformulation by retrieval performance:
- `max_oracle_nugget_all` - Best by nugget all score
- `max_oracle_nugget_strict_all_score` - Best by strict all score
- `max_oracle_nugget_strict_vital_score` - Best by strict vital score
- `max_oracle_nugget_vital_score` - Best by vital score
- `max_oracle_ndcg@10` - Best by NDCG@10
- `max_oracle_recall@100` - Best by Recall@100

### Max Oracle - Generation-Only (4 × 2 = 8 rows)
Select best reformulation by generation-only performance:
- **`max_oracle_genonly_nugget_all_score`** - Best by genonly all score
- **`max_oracle_genonly_nugget_strict_all_score`** - Best by genonly strict all score
- **`max_oracle_genonly_nugget_strict_vital_score`** - Best by genonly strict vital score
- **`max_oracle_genonly_nugget_vital_score`** - Best by genonly vital score

### Average Methods (6 × 2 = 12 rows)
Average across trials for each reformulation method:
- `avg_genqr`, `avg_genqr_ensemble`, `avg_mugi`
- `avg_qa_expand`, `avg_query2doc`, `avg_query2e`

### Original (1 × 2 = 2 rows)
- `original` - Original query performance

---

## 🔍 Key Insights

### Best Overall Performance
- **Cohere max_oracle_nugget_all**: 0.5332 (retrieval), 0.3838 (genonly)
- **Pyserini max_oracle_nugget_all**: 0.5107 (retrieval), 0.3928 (genonly)

### Generation-Only Oracles
When selecting by generation-only scores:
- **Genonly all score**: 0.5256 (both methods)
- **Retrieval performance**: 0.3572 (pyserini), 0.3991 (cohere)

This shows that reformulations optimized for generation-only are **different** from those optimized for retrieval!

### Retrieval vs Generation-Only
- Retrieval-based oracles achieve **higher retrieval performance**
- Generation-only oracles achieve **higher generation-only performance**
- Different reformulations are optimal for different scenarios

---

## 💡 Usage

```python
import pandas as pd

# Load filtered oracle data
df = pd.read_csv('qpp_oracle_analysis/final/qpp_oracle_performance_filtered.csv')

# Compare retrieval vs generation-only
df['improvement'] = (df['nugget_all_score_mean'] - df['genonly_all_score_mean']) / df['genonly_all_score_mean'] * 100

# Find best QPP methods for each retrieval system
best_pys = df[df['retrieval_method'] == 'pyserini'].nlargest(10, 'nugget_all_score_mean')
best_coh = df[df['retrieval_method'] == 'cohere'].nlargest(10, 'nugget_all_score_mean')

# Compare oracle types
oracles = df[df['qpp_metric'].str.contains('max_oracle', na=False)]
print(oracles[['qpp_metric', 'retrieval_method', 'nugget_all_score_mean', 'genonly_all_score_mean']])
```

---

## ✅ Summary

This filtered file provides a clean, focused view of:
- ✅ **8 key columns** (no std/n clutter)
- ✅ **Both retrieval methods** in one file
- ✅ **All oracle types** (retrieval-based and generation-only)
- ✅ **Direct comparison** between retrieval and generation-only performance

Perfect for analysis and visualization!
