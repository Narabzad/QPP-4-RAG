# QPP-4-RAG: Query Performance Prediction for Retrieval-Augmented Generation

This repository contains the code and results for the paper **"Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines"** (SIGIR 2026).

## Paper Abstract

Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? 

We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval, and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination—selecting the optimal reformulation among competing variants of the same information need.

## Repository Structure

This repository is organized into four main pipeline steps:

```
.
├── README.md                          # This file
├── querygym/                          # Main experimental directory
│   ├── queries/                      # Step 1: Query generation results
│   ├── rag_prepared/                 # Step 2: RAG prepared files
│   ├── rag_results/                  # Step 2: RAG generation results
│   ├── rag_nuggetized_eval/          # Step 3: Nuggetizer evaluation results
│   ├── qpp/                          # Step 4: QPP prediction files
│   │   ├── bert_qpp_results/         # Step 4: BERT-QPP predictions
│   │   └── QSDQPP/                   # Step 4: QSDQPP predictions
│   └── [scripts]                      # Additional scripts for evaluation, analysis, plotting, etc.
├── QPP4CS/                           # QPP method implementations
├── BERTQPP/                          # BERT-QPP implementation
└── nuggetizer/                       # Nuggetizer evaluation tool
```

## Pipeline Overview

The complete pipeline consists of four main steps:

### Step 1: Query Generation and Preparation

**Location**: `querygym/queries/`

This step generates query variants using multiple reformulation methods:
- **Original queries**: Baseline queries
- **6 reformulation methods**: `genqr`, `genqr_ensemble`, `mugi`, `qa_expand`, `query2doc`, `query2e`
- **5 trials per method**: Each method generates 5 different sets of reformulations
- **Total**: 31 query files (1 original + 30 reformulated)

**Query Files Format**:
```
<query_id>\t<query_text>
```

**Files**:
- `topics.original.txt` - Original queries
- `topics.genqr_trial1.txt` through `topics.genqr_trial5.txt`
- `topics.genqr_ensemble_trial1.txt` through `topics.genqr_ensemble_trial5.txt`
- `topics.mugi_trial1.txt` through `topics.mugi_trial5.txt`
- `topics.qa_expand_trial1.txt` through `topics.qa_expand_trial5.txt`
- `topics.query2doc_trial1.txt` through `topics.query2doc_trial5.txt`
- `topics.query2e_trial1.txt` through `topics.query2e_trial5.txt`

**Scripts**:
- `querygym/generate_query_files.py` - Generate query files from reformulation data
- `scripts/generate_query_reformulations.py` - Generate query reformulations using GPT-4o
- `scripts/filter_all_queries.py` - Filter queries to only those with qrels

### Step 2: Running RAG on Queries

**Locations**: 
- `querygym/rag_prepared/` - Prepared files in Ragnarok format
- `querygym/rag_results/` - Generated RAG answers

This step executes the RAG pipeline on all query variants:
1. **Retrieval**: Documents are retrieved using two methods:
   - **Pyserini BM25**: Sparse retrieval
   - **Cohere Rerank**: Dense retrieval with reranking
2. **Preparation**: Retrieval results are converted to Ragnarok format
3. **Generation**: GPT-4o generates answers using top-k retrieved documents

**Directory Structure**:
```
rag_prepared/
├── retrieval/                        # Pyserini retrieval prepared files
│   ├── ragnarok_format_run.original.json
│   ├── ragnarok_format_run.genqr_trial1.json
│   └── ...
└── retrieval_cohere/                # Cohere retrieval prepared files
    ├── ragnarok_format_run.original.json
    └── ...

rag_results/
├── retrieval/                        # Pyserini RAG results
│   ├── run.original_gpt-4o.json
│   ├── run.genqr_trial1_gpt-4o.json
│   └── ...
└── retrieval_cohere/                # Cohere RAG results
    ├── run.original_gpt-4o.json
    └── ...
```

**Scripts**:
- `querygym/run_RAG_on_prepared_files.py` - Execute RAG generation on prepared files

**Note**: RRF (Reciprocal Rank Fusion) results are **not included** in this repository as per paper requirements.

### Step 3: Running Nuggetizer Evaluation

**Location**: `querygym/rag_nuggetized_eval/`

This step evaluates RAG-generated answers using the Nuggetizer framework, which measures answer quality by checking support for information nuggets (key facts that should be present in answers).

**Directory Structure**:
```
rag_nuggetized_eval/
├── retrieval/                       # Pyserini evaluation results
│   ├── assignments/                  # Nugget assignment files
│   └── scores/                       # Nugget score files
└── retrieval_cohere/                # Cohere evaluation results
    ├── assignments/
    └── scores/
```

**Metrics Computed**:
- `strict_vital_score`: Strict support for vital nuggets
- `strict_all_score`: Strict support for all nuggets
- `vital_score`: Support for vital nuggets (with partial credit)
- `all_score`: Support for all nuggets (with partial credit)

**Scripts**:
- `querygym/run_rag_nuggetizer.py` - Run nuggetizer evaluation on RAG results

### Step 4: Running QPP Predictions

**Locations**:
- `querygym/qpp/` - Pre- and post-retrieval QPP metrics
- `querygym/qpp/bert_qpp_results/` - BERT-QPP predictions
- `querygym/qpp/QSDQPP/` - QSDQPP predictions (if available)

This step computes Query Performance Predictions using multiple methods:

#### 4.1 Pre-Retrieval QPP Methods

**Location**: `querygym/qpp/pre_retrieval_*.csv`

**Methods Included**:
- `ql`: Query Length
- `IDF-avg`, `IDF-max`, `IDF-sum`: Inverse Document Frequency statistics
- `SCQ-avg`, `SCQ-max`, `SCQ-sum`: Simplified Clarity Query metrics
- `avgICTF`: Average Inverse Collection Term Frequency
- `SCS-APX`, `SCS-FULL`: Simplified Clarity Score variants (approximate and full)
- `qsdqpp_predicted_ndcg`: QSDQPP predicted nDCG (from QSDQPP method)

**File Format**: CSV with columns: `query_id,ql,IDF-avg,IDF-max,...`

#### 4.2 Post-Retrieval QPP Methods

**Location**: `querygym/qpp/post_retrieval_*.csv`

**Methods Included**:
- `clarity-score-k100`: Clarity score at k=100
- `wig-norm-k100`, `wig-no-norm-k100`: Weighted Information Gain (normalized and non-normalized) at k=100
- `nqc-norm-k100`, `nqc-no-norm-k100`: Normalized Query Clarity at k=100
- `smv-norm-k100`, `smv-no-norm-k100`: Score Magnitude Variance
- `sigma-x0.5`, `sigma-max`: Score distribution statistics
- `RSD`: Robust Standard Deviation
- `qsdqpp_predicted_ndcg`: QSDQPP predicted nDCG (also available as post-retrieval)
- **BERT-QPP**: Cross-Encoder and Bi-Encoder BERT-QPP predictions (located in `querygym/qpp/bert_qpp_results/`)
- **QSDQPP**: QSDQPP predicted nDCG scores (standalone files in `querygym/qpp/QSDQPP/`)

**File Format**: CSV with columns: `query_id,clarity-score-k100,...` for most methods. QSDQPP also provides standalone TSV files: `topics.{method}_trial{trial}_predicted_ndcg.txt`

**Note**: Separate files for each retrieval method (pyserini, cohere). BERT-QPP predictions are available in JSON and CSV formats in `querygym/qpp/bert_qpp_results/`. QSDQPP is available both as a pre-retrieval metric (in pre-retrieval CSV files) and as a post-retrieval metric (in post-retrieval CSV files), with standalone prediction files in `querygym/qpp/QSDQPP/`.

**Scripts**:
- `querygym/run_bert_qpp.py` - Compute BERT-QPP predictions

#### 4.3 Other QPP Methods

Additional QPP implementations are available in:
- `QPP4CS/` - Contains implementations of various QPP methods
- `BERTQPP/` - Standalone BERT-QPP implementation

## Reproducing the Results

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **Java**: Required for Pyserini
3. **API Keys**:
   - OpenAI API key (set as `OPENAI_API_KEY`)
   - Cohere API key (set as `COHERE_API_KEY` or `CO_API_KEY`)

### Step-by-Step Reproduction

#### Step 1: Generate Query Files

```bash
cd querygym
python generate_query_files.py
```

This generates all query variant files in `queries/` directory.

#### Step 2: Run Retrieval and RAG

```bash
# Retrieve documents (Pyserini)
python retrieve_all_queries.py --k 100

# Retrieve documents (Cohere)
python retrieve_all_queries_cohere.py --k 100

# Convert to Ragnarok format
python scripts/convert_to_ragnarok_format.py \
    --queries querygym/queries/topics.original.txt \
    --results-file querygym/retrieval/run.pyserini.txt \
    --output-file querygym/rag_prepared/retrieval/ragnarok_format.json \
    --k 5

# Run RAG generation
python run_RAG_on_prepared_files.py \
    --input-dir rag_prepared/retrieval \
    --output-dir rag_results/retrieval \
    --model gpt-4o \
    --topk 5

# Run Nuggetizer evaluation
python scripts/run_nuggetizer_pipeline.py \
    --ragnarok-dir rag_results/retrieval/ \
    --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
    --assignments-dir rag_nuggetized_eval/retrieval/assignments/ \
    --scores-dir rag_nuggetized_eval/retrieval/scores/
```

#### Step 3: Run Nuggetizer Evaluation

```bash
python run_rag_nuggetizer.py \
    --rag-results-dir rag_results/retrieval \
    --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
    --output-dir rag_nuggetized_eval/retrieval \
    --model gpt-4o
```

#### Step 4: Compute QPP Predictions

```bash
# Pre-retrieval QPP
cd qpp
python run_pre_retrieval_verbose.py \
    --queries-dir ../queries \
    --output-dir . \
    --index msmarco-v2.1-doc-segmented

# Post-retrieval QPP (including BERT-QPP)
python run_qpp_querygym.py \
    --queries-dir ../queries \
    --runs-dir ../retrieval \
    --output-dir . \
    --index msmarco-v2.1-doc-segmented

# BERT-QPP (post-retrieval supervised method)
cd ..
python run_bert_qpp.py \
    --queries-dir queries \
    --retrieval-dir retrieval \
    --output-dir qpp/bert_qpp_results
```

## Results Summary

### Query Variants
- **Total queries**: 56 original queries
- **Reformulation methods**: 6 methods × 5 trials = 30 variant sets
- **Total query variants**: 1,736 (56 × 31)

### QPP Methods Evaluated
- **Pre-retrieval**: 11+ metrics
- **Post-retrieval**: 13+ metrics per retrieval method
- **BERT-QPP**: 2 variants (cross-encoder, bi-encoder)
- **Total QPP methods**: 37+ unique predictors

### Evaluation Metrics
- **Retrieval metrics**: nDCG@5, nDCG@10, MAP, etc.
- **Generation metrics**: Nugget scores (strict_vital, strict_all, vital, all)

### Oracle Analysis
Oracle performance analysis measures the best possible performance achievable by selecting query variants based on QPP predictions. See [`querygym/HOW_TO_COMPUTE_ORACLE.md`](querygym/HOW_TO_COMPUTE_ORACLE.md) for detailed instructions on computing oracle results.

## Key Findings

1. **Utility Gap**: Variants that maximize ranking metrics (nDCG) often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity.

2. **QPP Effectiveness**: QPP can reliably identify variants that improve end-to-end answer quality over the original query.

3. **Pre-retrieval Efficiency**: Lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG.

## Acknowledgments

This work uses:
- [Ragnarok](https://github.com/castorini/ragnarok) for RAG pipeline
- [Nuggetizer](https://github.com/castorini/nuggetizer) for answer evaluation
- [Pyserini](https://github.com/castorini/pyserini) for retrieval
- [QPP4CS](https://github.com/ChuanMeng/QPP4CS) for QPP implementations
