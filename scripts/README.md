# Scripts Directory

This directory contains utility scripts for the QPP-4-RAG pipeline. These scripts support the main pipeline steps described in the repository README.

## Scripts by Pipeline Step

### Step 1: Query Generation

- **`generate_query_reformulations.py`**: Generates query reformulations using GPT-4o. Reads queries from topics files and generates variations following specific instructions.

### Step 2: RAG Preparation

- **`retrieve_topics.py`**: Retrieves documents for queries in topics format files using Pyserini.
- **`run_pyserini_with_java_setup.py`**: Helper script to run Pyserini with proper Java environment setup.
- **`convert_to_ragnarok_format.py`**: Converts retrieved results to Ragnarok format for RAG generation.
- **`filter_all_queries.py`**: Filters query variation files to only include queries that have qrels (preprocessing utility).

### Step 3: Nuggetizer Evaluation

- **`run_nuggetizer_pipeline.py`**: Wrapper script to run nuggetizer assignment and scoring for Ragnarok results.
- **`aggregate_nuggetizer_scores.py`**: Aggregates nuggetizer scores across all prediction folders and calculates averages for the 4 metrics.

### Step 4: QPP Analysis

- **`merge_qpp_with_nuggets.py`**: Merges QPP metrics files with nugget scores for analysis.

## Usage Examples

### Generate Query Reformulations

```bash
python scripts/generate_query_reformulations.py \
    --input-file querygym/queries/topics.original.txt \
    --output-dir querygym/queries/
```

### Retrieve Documents

```bash
python scripts/retrieve_topics.py \
    --topics-file querygym/queries/topics.original.txt \
    --output-dir querygym/retrieval/ \
    --run-name pyserini \
    --k 100
```

### Convert to Ragnarok Format

```bash
python scripts/convert_to_ragnarok_format.py \
    --queries querygym/queries/topics.original.txt \
    --results-file querygym/retrieval/run.pyserini.txt \
    --output-file querygym/rag_prepared/retrieval/ragnarok_format.json \
    --k 5
```

### Run Nuggetizer Pipeline

```bash
python scripts/run_nuggetizer_pipeline.py \
    --ragnarok-dir querygym/rag_results/retrieval/ \
    --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
    --assignments-dir querygym/rag_nuggetized_eval/retrieval/assignments/ \
    --scores-dir querygym/rag_nuggetized_eval/retrieval/scores/
```

### Merge QPP with Nuggets

```bash
python scripts/merge_qpp_with_nuggets.py \
    --qpp-file querygym/qpp/post_retrieval_pyserini.csv \
    --nugget-scores querygym/rag_nuggetized_eval/retrieval/scores/scores.jsonl \
    --output-file querygym/qpp/post_retrieval_pyserini_with_nuggets.csv
```

## Notes

- Scripts may require specific environment variables (e.g., `SEWON_OPENAI_API_KEY` for OpenAI API calls).
- Some scripts require Java/Pyserini setup - see `run_pyserini_with_java_setup.py` for Java environment configuration.
- Paths in scripts may need to be adjusted based on your local setup.
- Scripts are provided as-is for reproducibility; users may need to modify paths and configurations.

## Excluded Scripts

The following scripts are **not** included as they are either:
- RRF-related (not part of the main pipeline)
- Test/verification scripts
- Training data generation scripts (not part of evaluation pipeline)
- Redundant with other included scripts

Excluded:
- `run_post_rrf_pipeline.py`
- `generate_rrf_variations.py`
- `run_prediction_pipeline_w_RRF.py`
- `rrf_predictions.py`
- `random_RRF.py`
- `run_batch_prediction_pipeline.py`
- `convert_predictions_to_ragnarok.py`
- `retrieve_predictions.py`
- `test_pyserini.py`
- `create_chatml_training_data.py`
- `make_training_data.py`
- `add_labeling_strategies.py`
- `assign_nuggets_incremental.py`
- `run_assign_score_nuggetizer.py`
- `score_all_nugget_assignments.py`
- `create_nuggets_from_rag_results.py`
- `generate_nuggets_from_qrels.py`
