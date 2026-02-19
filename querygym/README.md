# Qgym Query Retrieval Pipeline

This directory contains scripts to generate query files from Qgym reformulation results and retrieve documents using pyserini BM25.

## Overview

The pipeline processes queries from `/future/u/negara/home/set_based_QPP/data/Qgym_reformulation_results.json`:
- **1680 reformulated queries**: 56 original queries × 6 methods × 5 trials
- **56 original queries**: Unique queries for baseline comparison

## Files Generated

### Query Files (31 total)
- **30 reformulated query files**: 6 methods × 5 trials
  - Methods: `genqr`, `genqr_ensemble`, `mugi`, `qa_expand`, `query2doc`, `query2e`
  - Trials: 1-5 for each method
  - Format: `topics.{method}_trial{trial}.txt`
- **1 original query file**: `topics.original.txt`

### Retrieval Results (31 run files per retrieval method)
- **BM25 (pyserini)**: Format: `run.{method}_trial{trial}.txt` or `run.original.txt`
- **Cohere rerank**: Format: `run.{method}_trial{trial}.txt` or `run.original.txt`
- Each file contains TREC-format retrieval results
- K=100 documents per query
- Cohere method: BM25 top-1000 → Cohere rerank → top-100

## Scripts

### 1. `generate_query_files.py`
Generates 31 query files in topics format from the JSON reformulation results.

**Usage:**
```bash
python3 generate_query_files.py
```

**Output:**
- Query files saved to: `querygym/queries/`

### 2. `retrieve_all_queries.py`
Retrieves documents for all query files using pyserini BM25.

**Usage:**
```bash
python3 retrieve_all_queries.py --k 100
```

**Options:**
- `--queries-dir`: Directory containing query files (default: `querygym/queries`)
- `--output-dir`: Output directory for retrieval results (default: `querygym/retrieval`)
- `--k`: Number of documents to retrieve per query (default: 100)

**Output:**
- Retrieval results saved to: `querygym/retrieval/`

### 3. `retrieve_all_queries_cohere.py`
Retrieves documents for all query files using Cohere rerank API.
First retrieves candidates with BM25 (pyserini), then reranks with Cohere.

**Usage:**
```bash
python3 retrieve_all_queries_cohere.py --k 100
```

**Options:**
- `--queries-dir`: Directory containing query files (default: `querygym/queries`)
- `--output-dir`: Output directory for retrieval results (default: `querygym/retrieval_cohere`)
- `--k`: Number of documents to retrieve per query after reranking (default: 100)
- `--candidate-k`: Number of candidates to retrieve with BM25 before reranking (default: 1000)
- `--cohere-model`: Cohere rerank model to use (default: `rerank-english-v3.0`)

**Requirements:**
- Cohere API key set in environment variable `COHERE_API_KEY` or `CO_API_KEY`, or in `.env.local` file
- Install Cohere: `pip install cohere`

**Output:**
- Retrieval results saved to: `querygym/retrieval_cohere/`

### 4. `run_all.sh`
Runs the complete pipeline: generates query files and runs retrieval with pyserini BM25.

**Usage:**
```bash
bash run_all.sh
```

## Directory Structure

```
querygym/
├── generate_query_files.py           # Generate query files from JSON
├── retrieve_all_queries.py          # Retrieve documents using pyserini BM25
├── retrieve_all_queries_cohere.py   # Retrieve documents using Cohere rerank
├── run_all.sh                        # Run complete pipeline (BM25)
├── README.md                  # This file
├── queries/                   # Generated query files (31 files)
│   ├── topics.genqr_trial1.txt
│   ├── topics.genqr_trial2.txt
│   ├── ...
│   └── topics.original.txt
└── retrieval/                 # Retrieval results (31 run files)
    ├── run.genqr_trial1.txt
    ├── run.genqr_trial2.txt
    ├── ...
    └── run.original.txt
```

## Requirements

- Python 3
- pyserini (with Java environment configured)
- Access to `msmarco-v2.1-doc-segmented` index

## Notes

- Each query file contains 56 queries (one per original query)
- **BM25 retrieval**: Uses pyserini BM25 with K=100 documents per query
- **Cohere retrieval**: Uses BM25 (top-1000 candidates) → Cohere rerank → top-100 documents
- Results are in TREC format: `query_id Q0 doc_id rank score run_name`
- The Java environment is automatically configured in the retrieval scripts
- Cohere API key must be set (see `retrieve_all_queries_cohere.py` requirements)

