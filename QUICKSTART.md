# Quick Start Guide

This guide provides a quick overview of how to use this repository.

## Repository Overview

This repository contains code and results for evaluating Query Performance Prediction (QPP) methods for selecting query variants in RAG pipelines.

## Directory Structure

```
.
├── README.md                    # Main documentation
├── REPRODUCIBILITY.md           # Detailed reproduction instructions
├── FILES_INVENTORY.md           # Complete file listing
├── QUICKSTART.md                # This file
│
├── querygym/                    # Main experimental directory
│   ├── queries/                 # Step 1: Query variants (31 files)
│   ├── rag_prepared/            # Step 2: RAG prepared files
│   ├── rag_results/            # Step 2: RAG generated answers
│   ├── rag_nuggetized_eval/     # Step 3: Nuggetizer evaluation
│   ├── qpp/                     # Step 4: QPP predictions (93 CSV files)
│   │   ├── bert_qpp_results/    # Step 4: BERT-QPP predictions
│   │   └── QSDQPP/              # Step 4: QSDQPP predictions
│
├── QPP4CS/                      # QPP method implementations
├── BERTQPP/                     # BERT-QPP implementation
└── scripts/                     # Utility scripts
```

## Quick Access to Results

### Query Variants
```bash
ls querygym/queries/*.txt
# 31 files: original + 6 methods × 5 trials
```

### RAG Results
```bash
# Pyserini results
ls querygym/rag_results/retrieval/*.json
# 31 files

# Cohere results
ls querygym/rag_results/retrieval_cohere/*.json
# 31 files
```

### Nuggetizer Scores
```bash
# Pyserini scores
ls querygym/rag_nuggetized_eval/retrieval/scores/*.jsonl
# 31 files

# Cohere scores
ls querygym/rag_nuggetized_eval/retrieval_cohere/scores/*.jsonl
# 31 files
```

### QPP Predictions
```bash
# Pre-retrieval QPP
ls querygym/qpp/pre_retrieval_*.csv
# 31 files

# Post-retrieval QPP
ls querygym/qpp/post_retrieval_*.csv
# 62 files (31 × 2 retrieval methods)

# BERT-QPP
ls querygym/qpp/bert_qpp_results/*.json
# 1 file (contains all predictions)

# QSDQPP
ls querygym/qpp/QSDQPP/*_predicted_ndcg.txt
# 31 files
```

## Understanding the Results

### Query Variants
Each query variant file contains 56 queries in TSV format:
```
2024-105741	is it dangerous to have wbc over 15,000 without treatment?
2024-121840	should teachers notify parents about state testing?
...
```

### RAG Results
Each RAG result file is a JSON array with generated answers:
```json
[
  {
    "topic_id": "2024-105741",
    "topic": "is it dangerous to have wbc over 15,000 without treatment?",
    "answer": [...],
    "references": [...]
  },
  ...
]
```

### Nuggetizer Scores
Each score file contains JSONL with nugget metrics:
```jsonl
{"qid": "2024-105741", "strict_vital_score": 0.5, "strict_all_score": 0.4, "vital_score": 0.6, "all_score": 0.5}
{"qid": "2024-121840", "strict_vital_score": 0.7, ...}
...
```

### QPP Predictions
Each CSV file contains QPP metrics per query:
```csv
query_id,ql,IDF-avg,IDF-max,clarity-score-k100,...
2024-105741,12,5.626,8.234,0.123,...
2024-121840,8,4.521,7.891,0.145,...
...
```

## Key Metrics

### Nuggetizer Metrics
- **strict_vital_score**: Strict support for vital nuggets (0-1)
- **strict_all_score**: Strict support for all nuggets (0-1)
- **vital_score**: Support for vital nuggets with partial credit (0-1)
- **all_score**: Support for all nuggets with partial credit (0-1)

### QPP Metrics
- **Pre-retrieval**: ql, IDF-*, SCQ-*, avgICTF, SCS-APX, SCS-FULL
- **Post-retrieval**: clarity-score-k100, wig-norm-k100, wig-no-norm-k100, nqc-norm-k100, nqc-no-norm-k100, smv-*, sigma-*, RSD
- **BERT-QPP**: Cross-encoder and bi-encoder predictions
- **QSDQPP**: Predicted nDCG scores

## Common Tasks

### 1. Find Best Query Variant for a Query

```python
import pandas as pd
import json

# Load nuggetizer scores
scores = []
for file in glob.glob("querygym/rag_nuggetized_eval/retrieval/scores/*.jsonl"):
    with open(file) as f:
        for line in f:
            scores.append(json.loads(line))

# Find best variant for query 2024-105741
query_scores = [s for s in scores if s['qid'] == '2024-105741']
best = max(query_scores, key=lambda x: x['all_score'])
print(f"Best variant: {best['qid']} with score {best['all_score']}")
```

### 2. Correlate QPP with Nugget Scores

```python
import pandas as pd

# Load QPP predictions
qpp = pd.read_csv("querygym/qpp/pre_retrieval_original_qpp_metrics.csv")

# Load nugget scores
nuggets = []
with open("querygym/rag_nuggetized_eval/retrieval/scores/run.original_gpt-4o.jsonl") as f:
    for line in f:
        nuggets.append(json.loads(line))

# Merge and correlate
nuggets_df = pd.DataFrame(nuggets)
merged = qpp.merge(nuggets_df, left_on='query_id', right_on='qid')
correlation = merged['IDF-avg'].corr(merged['all_score'])
print(f"Correlation: {correlation:.3f}")
```

### 3. Compare QPP Methods

```python
import pandas as pd
import glob

# Load all pre-retrieval QPP files
qpp_files = glob.glob("querygym/qpp/pre_retrieval_*.csv")
all_qpp = []

for file in qpp_files:
    df = pd.read_csv(file)
    method = file.split('_')[2]  # Extract method name
    df['method'] = method
    all_qpp.append(df)

combined = pd.concat(all_qpp)
# Now you can compare methods
```

## Next Steps

1. **Read the paper**: Understand the research questions and methodology
2. **Explore results**: Browse the result files to understand the data
3. **Reproduce**: Follow `REPRODUCIBILITY.md` to run the full pipeline
4. **Analyze**: Use the provided scripts or create your own analysis

## Getting Help

- **Documentation**: See `README.md` for overview, `REPRODUCIBILITY.md` for detailed instructions
- **File listing**: See `FILES_INVENTORY.md` for complete file listing
- **Issues**: Open an issue on GitHub for questions or problems

