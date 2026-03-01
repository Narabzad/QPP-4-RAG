# Reproducibility Guide

This document provides detailed instructions for reproducing all results in the paper.

## Environment Setup

### 1. Python Environment

```bash
# Create conda environment
conda create -n qpp4rag python=3.8
conda activate qpp4rag

# Install core dependencies
pip install torch transformers sentence-transformers
pip install pyserini
pip install openai cohere
pip install tqdm pandas numpy
```

### 2. Java Setup (for Pyserini)

Pyserini requires Java 11+. Install it via your package manager or conda:

```bash
# Option 1: conda
conda install -c conda-forge openjdk=21

# Option 2: system package manager (Ubuntu/Debian)
sudo apt-get install -y openjdk-21-jdk

# Then export JAVA_HOME (adjust path to your installation):
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
# Or, if using conda:
export JAVA_HOME=$CONDA_PREFIX
```

### 3. API Keys

```bash
# OpenAI API key
export OPENAI_API_KEY="your-openai-key"

# Cohere API key
export COHERE_API_KEY="your-cohere-key"
# OR
export CO_API_KEY="your-cohere-key"
```

### 4. Data Requirements

- **TREC-RAG Dataset**: TREC 2024 RAG Track test queries and qrels
- **MS MARCO v2.1 Document Index**: For retrieval (downloaded automatically by Pyserini)
- **Nugget File**: `data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl`

## Complete Pipeline Execution

### Step 1: Query Generation

**Input**: Reformulation data (JSON format with query variants)

**Script**: `querygym/generate_query_files.py`

```bash
cd querygym
python generate_query_files.py
```

**Output**: 
- 31 query files in `queries/` directory
- Format: `topics.{method}_trial{trial}.txt` or `topics.original.txt`

**Expected Output**:
- `queries/topics.original.txt` (56 queries)
- `queries/topics.genqr_trial1.txt` through `topics.genqr_trial5.txt` (5 files)
- `queries/topics.genqr_ensemble_trial1.txt` through `topics.genqr_ensemble_trial5.txt` (5 files)
- `queries/topics.mugi_trial1.txt` through `topics.mugi_trial5.txt` (5 files)
- `queries/topics.qa_expand_trial1.txt` through `topics.qa_expand_trial5.txt` (5 files)
- `queries/topics.query2doc_trial1.txt` through `topics.query2doc_trial5.txt` (5 files)
- `queries/topics.query2e_trial1.txt` through `topics.query2e_trial5.txt` (5 files)

**Total**: 31 files

### Step 2: Retrieval

#### 2.1 Pyserini BM25 Retrieval

**Script**: `querygym/retrieve_all_queries.py`

```bash
python retrieve_all_queries.py \
    --queries-dir queries \
    --output-dir retrieval \
    --k 100
```

**Output**: 
- 31 TREC-format run files in `retrieval/`
- Format: `run.{method}_trial{trial}.txt` or `run.original.txt`

#### 2.2 Cohere Rerank Retrieval

**Script**: `querygym/retrieve_all_queries_cohere.py`

```bash
python retrieve_all_queries_cohere.py \
    --queries-dir queries \
    --output-dir retrieval_cohere \
    --k 100 \
    --candidate-k 1000
```

**Output**:
- 31 TREC-format run files in `retrieval_cohere/`

### Step 3: RAG Preparation

**Script**: `scripts/convert_to_ragnarok_format.py`

```bash
# Run from the repository root.
# Convert each of the 31 retrieval run files to Ragnarok format.
# Example for pyserini results:
for run_file in querygym/retrieval/run.*.txt; do
    python scripts/convert_to_ragnarok_format.py \
        --queries querygym/queries/topics.original.txt \
        --single-file "$run_file" \
        --output-dir querygym/rag_prepared/retrieval \
        --k 5
done

# For Cohere results:
for run_file in querygym/retrieval_cohere/run.*.txt; do
    python scripts/convert_to_ragnarok_format.py \
        --queries querygym/queries/topics.original.txt \
        --single-file "$run_file" \
        --output-dir querygym/rag_prepared/retrieval_cohere \
        --k 5
done
```

**Output**:
- 31 JSON files in `rag_prepared/retrieval/`
- 31 JSON files in `rag_prepared/retrieval_cohere/`

**Note**: This step converts TREC-format retrieval results to Ragnarok format for RAG generation.

### Step 4: RAG Generation

**Script**: `querygym/run_RAG_on_prepared_files.py`

```bash
cd querygym
# Process all prepared files (both retrieval methods) with default settings
python run_RAG_on_prepared_files.py \
    --input-dirs rag_prepared/retrieval_cohere rag_prepared/retrieval \
    --output-dirs rag_results/retrieval_cohere rag_results/retrieval \
    --model gpt-4o \
    --topk 3 \
    --num-workers 8
```

**Output**:
- 31 JSON files in `rag_results/retrieval/`
- 31 JSON files in `rag_results/retrieval_cohere/`

**Format**: Each file contains RAG-generated answers with citations.

### Step 5: Nuggetizer Evaluation

**Script**: `querygym/run_rag_nuggetizer.py`

```bash
# For Pyserini results
python run_rag_nuggetizer.py \
    --rag-results-dir rag_results/retrieval \
    --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
    --output-dir rag_nuggetized_eval/retrieval \
    --model gpt-4o

# For Cohere results
python run_rag_nuggetizer.py \
    --rag-results-dir rag_results/retrieval_cohere \
    --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
    --output-dir rag_nuggetized_eval/retrieval_cohere \
    --model gpt-4o
```

**Output**:
- Assignment files in `rag_nuggetized_eval/retrieval/assignments/`
- Score files in `rag_nuggetized_eval/retrieval/scores/`
- Same structure for `retrieval_cohere/`

**Metrics**: Each score file contains:
- `strict_vital_score`
- `strict_all_score`
- `vital_score`
- `all_score`

### Step 6: QPP Predictions

#### 6.1 Pre-Retrieval QPP

**Script**: `querygym/qpp/run_pre_retrieval_verbose.py`

```bash
cd querygym/qpp
python run_pre_retrieval_verbose.py \
    --queries-dir ../queries \
    --output-dir . \
    --index msmarco-v2.1-doc-segmented
```

**Output**:
- 31 CSV files: `pre_retrieval_{method}_trial{trial}_qpp_metrics.csv`
- 1 file: `pre_retrieval_original_qpp_metrics.csv`

**Metrics Included**:
- `ql`: Query Length
- `IDF-avg`, `IDF-max`, `IDF-sum`
- `SCQ-avg`, `SCQ-max`, `SCQ-sum`
- `avgICTF`
- `SCS-APX`, `SCS-FULL` (Simplified Clarity Score - approximate and full)

#### 6.2 Post-Retrieval QPP

**Script**: `querygym/qpp/run_qpp_querygym.py`

```bash
cd querygym/qpp
# Run post-retrieval QPP for both retrieval methods simultaneously
python run_qpp_querygym.py \
    --queries_dir ../queries \
    --retrieval_dirs ../retrieval ../retrieval_cohere \
    --output_dir . \
    --index_path msmarco-v2.1-doc-segmented \
    --mode post
```

**Output**:
- 62 CSV files: `post_retrieval_{method}_trial{trial}_{method}_trial{trial}_{retrieval_method}_qpp_metrics.csv`
- 2 files: `post_retrieval_original_original_{retrieval_method}_qpp_metrics.csv`

**Metrics Included**:
- `clarity-score-k100`
- `wig-norm-k100`, `wig-no-norm-k100`
- `nqc-norm-k100`, `nqc-no-norm-k100`
- `smv-norm-k100`, `smv-no-norm-k100`
- `sigma-x0.5`, `sigma-max`
- `RSD`
- `qsdqpp_predicted_ndcg`

#### 6.3 BERT-QPP Predictions

**Script**: `querygym/run_bert_qpp.py`

```bash
cd ..
python run_bert_qpp.py \
    --queries-dir queries \
    --retrieval-dir retrieval \
    --retrieval-cohere-dir retrieval_cohere \
    --output-dir qpp/bert_qpp_results \
    --ce-model-path path/to/cross-encoder-model \
    --bi-model-path path/to/bi-encoder-model
```

**Output**:
- `qpp/bert_qpp_results/bert_qpp_scores.json`
- `qpp/bert_qpp_results/bert_qpp_scores.csv`

**Note**: BERT-QPP models need to be trained separately or downloaded. See `BERTQPP/README.md` for details.

## Verification

After running all steps, verify the outputs:

### Expected File Counts

- **Query files**: 31 files in `queries/`
- **Retrieval results**: 31 files in `retrieval/`, 31 in `retrieval_cohere/`
- **RAG prepared**: 31 files in `rag_prepared/retrieval/`, 31 in `rag_prepared/retrieval_cohere/`
- **RAG results**: 31 files in `rag_results/retrieval/`, 31 in `rag_results/retrieval_cohere/`
- **Nuggetizer scores**: 31 score files in `rag_nuggetized_eval/retrieval/scores/`, 31 in `rag_nuggetized_eval/retrieval_cohere/scores/`
- **Pre-retrieval QPP**: 31 CSV files in `qpp/`
- **Post-retrieval QPP**: 62 CSV files in `qpp/` (31 × 2 retrieval methods)
- **BERT-QPP**: 2 files in `qpp/bert_qpp_results/`

### Data Validation

```bash
# Check query files
ls querygym/queries/*.txt | wc -l  # Should be 31

# Check retrieval results
ls querygym/retrieval/*.txt | wc -l  # Should be 31
ls querygym/retrieval_cohere/*.txt | wc -l  # Should be 31

# Check RAG results
ls querygym/rag_results/retrieval/*.json | wc -l  # Should be 31
ls querygym/rag_results/retrieval_cohere/*.json | wc -l  # Should be 31

# Check QPP predictions
ls querygym/qpp/pre_retrieval_*.csv | wc -l  # Should be 31
ls querygym/qpp/post_retrieval_*.csv | wc -l  # Should be 62
```

## Troubleshooting

### Common Issues

1. **Java/Pyserini errors**: Ensure Java is properly configured and JVM_PATH is set
2. **API key errors**: Verify environment variables are set correctly
3. **Memory issues**: Reduce `--num-workers` or process files in batches
4. **Missing index**: Pyserini will download the index automatically on first use

### Performance Notes

- **RAG generation**: Most time-consuming step (~hours for full dataset)
- **QPP computation**: Pre-retrieval is fast, post-retrieval takes longer
- **Nuggetizer**: Moderate time, depends on API rate limits

## Additional Resources

- See `README.md` for overview
- See `querygym/README.md` for query-specific documentation
- See `QPP4CS/README.md` for QPP method details
- See `BERTQPP/README.md` for BERT-QPP training instructions
