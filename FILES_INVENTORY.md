# Files Inventory

This document lists all files that should be included in the repository for reproducibility.

## Step 1: Query Generation

### Query Files
**Location**: `querygym/queries/`

- `topics.original.txt` - Original queries (56 queries)
- `topics.genqr_trial1.txt` through `topics.genqr_trial5.txt` (5 files)
- `topics.genqr_ensemble_trial1.txt` through `topics.genqr_ensemble_trial5.txt` (5 files)
- `topics.mugi_trial1.txt` through `topics.mugi_trial5.txt` (5 files)
- `topics.qa_expand_trial1.txt` through `topics.qa_expand_trial5.txt` (5 files)
- `topics.query2doc_trial1.txt` through `topics.query2doc_trial5.txt` (5 files)
- `topics.query2e_trial1.txt` through `topics.query2e_trial5.txt` (5 files)

**Total**: 31 query files

### Scripts
- `querygym/generate_query_files.py` - Generate query files from reformulation data

## Step 2: RAG Execution

### RAG Prepared Files
**Location**: `querygym/rag_prepared/`

#### Pyserini Retrieval
- `retrieval/ragnarok_format_run.original.json`
- `retrieval/ragnarok_format_run.genqr_trial1.json` through `ragnarok_format_run.genqr_trial5.json` (5 files)
- `retrieval/ragnarok_format_run.genqr_ensemble_trial1.json` through `ragnarok_format_run.genqr_ensemble_trial5.json` (5 files)
- `retrieval/ragnarok_format_run.mugi_trial1.json` through `ragnarok_format_run.mugi_trial5.json` (5 files)
- `retrieval/ragnarok_format_run.qa_expand_trial1.json` through `ragnarok_format_run.qa_expand_trial5.json` (5 files)
- `retrieval/ragnarok_format_run.query2doc_trial1.json` through `ragnarok_format_run.query2doc_trial5.json` (5 files)
- `retrieval/ragnarok_format_run.query2e_trial1.json` through `ragnarok_format_run.query2e_trial5.json` (5 files)

**Total**: 31 files

#### Cohere Retrieval
- Same structure in `retrieval_cohere/` directory
- **Total**: 31 files

### RAG Results
**Location**: `querygym/rag_results/`

#### Pyserini Results
- `retrieval/run.original_gpt-4o.json`
- `retrieval/run.genqr_trial1_gpt-4o.json` through `run.genqr_trial5_gpt-4o.json` (5 files)
- `retrieval/run.genqr_ensemble_trial1_gpt-4o.json` through `run.genqr_ensemble_trial5_gpt-4o.json` (5 files)
- `retrieval/run.mugi_trial1_gpt-4o.json` through `run.mugi_trial5_gpt-4o.json` (5 files)
- `retrieval/run.qa_expand_trial1_gpt-4o.json` through `run.qa_expand_trial5_gpt-4o.json` (5 files)
- `retrieval/run.query2doc_trial1_gpt-4o.json` through `run.query2doc_trial5_gpt-4o.json` (5 files)
- `retrieval/run.query2e_trial1_gpt-4o.json` through `run.query2e_trial5_gpt-4o.json` (5 files)

**Total**: 31 files

#### Cohere Results
- Same structure in `retrieval_cohere/` directory
- **Total**: 31 files

### Scripts
- `querygym/run_RAG_on_prepared_files.py` - Execute RAG generation

## Step 3: Nuggetizer Evaluation

### Nuggetizer Results
**Location**: `querygym/rag_nuggetized_eval/`

#### Pyserini Results
- `retrieval/assignments/` - Nugget assignment files (31 files)
- `retrieval/scores/` - Nugget score files (31 files)

#### Cohere Results
- `retrieval_cohere/assignments/` - Nugget assignment files (31 files)
- `retrieval_cohere/scores/` - Nugget score files (31 files)

**Total**: 124 files (31 assignments + 31 scores) × 2 retrieval methods

### Scripts
- `querygym/run_rag_nuggetizer.py` - Run nuggetizer evaluation

## Step 4: QPP Predictions

### Pre-Retrieval QPP
**Location**: `querygym/qpp/`

- `pre_retrieval_original_qpp_metrics.csv`
- `pre_retrieval_genqr_trial1_qpp_metrics.csv` through `pre_retrieval_genqr_trial5_qpp_metrics.csv` (5 files)
- `pre_retrieval_genqr_ensemble_trial1_qpp_metrics.csv` through `pre_retrieval_genqr_ensemble_trial5_qpp_metrics.csv` (5 files)
- `pre_retrieval_mugi_trial1_qpp_metrics.csv` through `pre_retrieval_mugi_trial5_qpp_metrics.csv` (5 files)
- `pre_retrieval_qa_expand_trial1_qpp_metrics.csv` through `pre_retrieval_qa_expand_trial5_qpp_metrics.csv` (5 files)
- `pre_retrieval_query2doc_trial1_qpp_metrics.csv` through `pre_retrieval_query2doc_trial5_qpp_metrics.csv` (5 files)
- `pre_retrieval_query2e_trial1_qpp_metrics.csv` through `pre_retrieval_query2e_trial5_qpp_metrics.csv` (5 files)

**Total**: 31 CSV files

**Metrics Included**:
- ql, IDF-avg, IDF-max, IDF-sum
- SCQ-avg, SCQ-max, SCQ-sum
- avgICTF, SCS-APX, SCS-FULL

### Post-Retrieval QPP
**Location**: `querygym/qpp/`

#### Pyserini Post-Retrieval QPP
- `post_retrieval_original_original_pyserini_qpp_metrics.csv`
- `post_retrieval_genqr_trial1_genqr_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_genqr_trial5_genqr_trial5_pyserini_qpp_metrics.csv` (5 files)
- `post_retrieval_genqr_ensemble_trial1_genqr_ensemble_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_genqr_ensemble_trial5_genqr_ensemble_trial5_pyserini_qpp_metrics.csv` (5 files)
- `post_retrieval_mugi_trial1_mugi_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_mugi_trial5_mugi_trial5_pyserini_qpp_metrics.csv` (5 files)
- `post_retrieval_qa_expand_trial1_qa_expand_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_qa_expand_trial5_qa_expand_trial5_pyserini_qpp_metrics.csv` (5 files)
- `post_retrieval_query2doc_trial1_query2doc_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_query2doc_trial5_query2doc_trial5_pyserini_qpp_metrics.csv` (5 files)
- `post_retrieval_query2e_trial1_query2e_trial1_pyserini_qpp_metrics.csv` through `post_retrieval_query2e_trial5_query2e_trial5_pyserini_qpp_metrics.csv` (5 files)

**Total**: 31 CSV files

#### Cohere Post-Retrieval QPP
- Same structure with `_cohere_` in filename
- **Total**: 31 CSV files

**Total Post-Retrieval QPP**: 62 CSV files (31 × 2 retrieval methods)

**Metrics Included**:
- clarity-score-k100
- wig-norm-k100, wig-no-norm-k100
- nqc-norm-k100, nqc-no-norm-k100
- smv-norm-k100, smv-no-norm-k100
- sigma-x0.5, sigma-max
- RSD

### BERT-QPP Predictions
**Location**: `querygym/qpp/bert_qpp_results/`

- `bert_qpp_scores.json` - JSON format with all predictions
- `bert_qpp_scores.csv` - CSV format for analysis

**Methods**:
- Cross-Encoder BERT-QPP
- Bi-Encoder BERT-QPP

### QSDQPP Predictions (if available)
**Location**: `querygym/qpp/QSDQPP/`

- `topics.original_predicted_ndcg.txt`
- `topics.genqr_trial1_predicted_ndcg.txt` through `topics.genqr_trial5_predicted_ndcg.txt` (5 files)
- `topics.genqr_ensemble_trial1_predicted_ndcg.txt` through `topics.genqr_ensemble_trial5_predicted_ndcg.txt` (5 files)
- `topics.mugi_trial1_predicted_ndcg.txt` through `topics.mugi_trial5_predicted_ndcg.txt` (5 files)
- `topics.qa_expand_trial1_predicted_ndcg.txt` through `topics.qa_expand_trial5_predicted_ndcg.txt` (5 files)
- `topics.query2doc_trial1_predicted_ndcg.txt` through `topics.query2doc_trial5_predicted_ndcg.txt` (5 files)
- `topics.query2e_trial1_predicted_ndcg.txt` through `topics.query2e_trial5_predicted_ndcg.txt` (5 files)

**Total**: 31 files (if available)

### QPP Scripts
- `querygym/qpp/run_pre_retrieval_verbose.py` - Compute pre-retrieval QPP
- `querygym/qpp/run_qpp_querygym.py` - Compute post-retrieval QPP
- `querygym/run_bert_qpp.py` - Compute BERT-QPP predictions
- `querygym/qpp/run_clarity_k100_and_wig.py` - Compute clarity and WIG metrics
- `querygym/qpp/run_clarity_only.py` - Compute clarity scores
- `querygym/qpp/run_clarity_from_text.py` - Compute clarity from text

## QPP Method Implementations

### QPP4CS
**Location**: `QPP4CS/`

Contains implementations of various QPP methods:
- `supervisedQPP/BERTQPP/` - BERT-QPP implementation
- `supervisedQPP/NQAQPP/` - NQAQPP implementation
- `supervisedQPP/qppBERTPL/` - qppBERTPL implementation
- `unsupervisedQPP/pre_retrieval.py` - Pre-retrieval QPP methods
- `unsupervisedQPP/post_retrieval.py` - Post-retrieval QPP methods

### BERTQPP
**Location**: `BERTQPP/`

Standalone BERT-QPP implementation with training and inference scripts.

## Summary

### File Counts

| Category | Count |
|----------|-------|
| Query files | 31 |
| RAG prepared (Pyserini) | 31 |
| RAG prepared (Cohere) | 31 |
| RAG results (Pyserini) | 31 |
| RAG results (Cohere) | 31 |
| Nuggetizer assignments | 62 |
| Nuggetizer scores | 62 |
| Pre-retrieval QPP | 31 |
| Post-retrieval QPP | 62 |
| BERT-QPP | 2 |
| QSDQPP (if available) | 31 |

**Total Data Files**: ~405 files

### Scripts

| Category | Scripts |
|----------|---------|
| Query generation | 1 |
| Retrieval | 2 |
| RAG preparation | 1 |
| RAG execution | 1 |
| Nuggetizer | 1 |
| QPP computation | 6+ |
| QPP implementations | Multiple in QPP4CS/ and BERTQPP/ |

## Notes

1. **RRF Results**: RRF (Reciprocal Rank Fusion) results are **NOT included** as per paper requirements.

2. **File Formats**:
   - Query files: TSV format (`query_id\tquery_text`)
   - Retrieval results: TREC format
   - RAG prepared: JSON (Ragnarok format)
   - RAG results: JSON
   - Nuggetizer: JSONL
   - QPP predictions: CSV

3. **File Sizes**: 
   - Query files: Small (~KB)
   - RAG results: Medium (~MB per file)
   - QPP predictions: Small (~KB per file)

4. **Missing Files**: If any files are missing, they can be regenerated using the provided scripts following the reproducibility guide.
