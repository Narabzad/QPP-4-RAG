# QPP Metrics for QueryGym Dataset

This directory contains scripts and results for running Query Performance Prediction (QPP) metrics on the QueryGym dataset.

## Script: `run_qpp_querygym.py`

A modular script to compute pre-retrieval and post-retrieval QPP metrics on QueryGym queries and retrieval results.

### Features

- **Pre-retrieval QPP metrics**: QS, PMI, VAR, IDF, SCQ, avgICTF, SCS-APX, SCS-FULL, query length
- **Post-retrieval QPP metrics**: CLARITY, WIG, NQC, SMV, SIGMA, RSD
- **Modular execution**: Run all, pre-retrieval only, or post-retrieval only
- **Parallel processing**: Configurable number of parallel workers
- **Flexible filtering**: Process specific query files

### Usage

#### Basic Usage

```bash
# Run all QPP methods (pre + post retrieval)
python run_qpp_querygym.py --mode all

# Run only pre-retrieval methods
python run_qpp_querygym.py --mode pre

# Run only post-retrieval methods
python run_qpp_querygym.py --mode post
```

#### Advanced Usage

```bash
# Run with custom parallelism
python run_qpp_querygym.py --mode all --num_processes 10

# Run on specific query files only
python run_qpp_querygym.py --mode all --query_files topics.original.txt topics.genqr_trial1.txt

# Custom directories
python run_qpp_querygym.py --mode all \
    --queries_dir /path/to/queries \
    --retrieval_dirs /path/to/retrieval1 /path/to/retrieval2 \
    --output_dir /path/to/output
```

### Command-Line Arguments

- `--mode`: Which QPP methods to run (`all`, `pre`, or `post`)
- `--num_processes`: Number of parallel processes (default: min(8, CPU count))
- `--queries_dir`: Directory containing query files (default: `querygym/queries`)
- `--retrieval_dirs`: Directories containing retrieval run files (default: `querygym/retrieval` and `querygym/retrieval_cohere`)
- `--index_path`: Pyserini prebuilt index name (default: `msmarco-v2.1-doc-segmented`)
- `--output_dir`: Output directory for QPP results (default: `querygym/qpp`)
- `--k_top`: Number of top retrieved items to process (default: 100)
- `--query_files`: Specific query files to process (optional)

### Input Files

**Query Files** (from `querygym/queries/`):
- Format: `topics.<name>.txt`
- Format: Tab-separated `query_id\tquery_text`

**Retrieval Files** (from `querygym/retrieval/` and `querygym/retrieval_cohere/`):
- Format: `run.<name>.txt`
- Format: TREC run format `query_id Q0 doc_id rank score run_name`

### Output Files

**Pre-retrieval metrics**:
- `pre_retrieval_<query_name>_qpp_metrics.csv`
- Contains: QS, PMI-avg, PMI-max, PMI-sum, ql, VAR-var-avg, VAR-var-max, VAR-var-sum, IDF-avg, IDF-max, IDF-sum, SCQ-avg, SCQ-max, SCQ-sum, avgICTF, SCS-APX, SCS-FULL

**Post-retrieval metrics**:
- `post_retrieval_<query_name>_<run_name>_<retrieval_method>_qpp_metrics.csv`
- Contains: clarity-score-k100, wig-norm-k100, wig-no-norm-k100, nqc-norm-k100, nqc-no-norm-k100, smv-norm-k100, smv-no-norm-k100, sigma-x0.5, sigma-max, RSD

### Example Workflow

```bash
# 1. Run all QPP methods
python run_qpp_querygym.py --mode all

# 2. Check results
ls -lh /future/u/negara/home/set_based_QPP/querygym/qpp/

# 3. Run only pre-retrieval for specific queries
python run_qpp_querygym.py --mode pre --query_files topics.original.txt

# 4. Run only post-retrieval with more parallelism
python run_qpp_querygym.py --mode post --num_processes 16
```

### Requirements

- Python 3.x
- pyserini
- numpy
- pytrec_eval
- tqdm
- Access to `msmarco-v2.1-doc-segmented` prebuilt index

### Notes

- The script uses multiprocessing for parallel execution
- Each query file is processed independently
- Results are saved immediately as each task completes
- The script handles missing run files gracefully
