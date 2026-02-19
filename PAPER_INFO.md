# Paper Information

## Title
**Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines**

## Conference
SIGIR 2026

## Abstract

Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? 

We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval, and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination—selecting the optimal reformulation among competing variants of the same information need. 

Through large-scale experiments on TREC-RAG using both sparse and dense retrievers, we evaluate pre- and post-retrieval predictors under correlation- and decision-based metrics. Our results reveal a systematic divergence between retrieval and generation objectives: variants that maximize ranking metrics such as nDCG often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity. Nevertheless, QPP can reliably identify variants that improve end-to-end answer quality over the original query. Notably, lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG.

## Key Contributions

1. **Intra-topic QPP evaluation**: First study of QPP for selecting among query variants of the same information need
2. **Utility gap discovery**: Systematic divergence between retrieval metrics and generation quality
3. **Comprehensive evaluation**: Large-scale experiments with multiple QPP methods, retrieval systems, and evaluation metrics
4. **Practical insights**: Pre-retrieval QPP methods are often sufficient for variant selection

## Dataset

- **TREC-RAG**: TREC 2024 RAG Track test set
- **Queries**: 56 test queries
- **Query variants**: 31 sets (1 original + 6 methods × 5 trials)
- **Total variants**: 1,736 query variants (56 queries × 31 sets)

## Query Reformulation Methods

1. **genqr**: Generative Query Reformulation
2. **genqr_ensemble**: Ensemble generative query reformulation
3. **mugi**: Multi-query generation
4. **qa_expand**: Question-answer expansion
5. **query2doc**: Query-to-document expansion
6. **query2e**: Query-to-entity expansion

Each method generates 5 different trial sets.

## Retrieval Methods

1. **Pyserini BM25**: Sparse retrieval using BM25
2. **Cohere Rerank**: Dense retrieval with Cohere reranking (BM25 top-1000 → Cohere rerank → top-100)

## QPP Methods Evaluated

### Pre-Retrieval QPP
- Query Length (ql)
- Inverse Document Frequency (IDF-avg, IDF-max, IDF-sum)
- Simplified Clarity Query (SCQ-avg, SCQ-max, SCQ-sum)
- Average Inverse Collection Term Frequency (avgICTF)
- Simplified Clarity Score (SCS-APX, SCS-FULL)
- QSDQPP predicted nDCG

### Post-Retrieval QPP
- Clarity Score (clarity-score-k100)
- Weighted Information Gain (wig-norm-k100, wig-no-norm-k100)
- Normalized Query Clarity (nqc-norm-k100, nqc-no-norm-k100)
- Score Magnitude Variance (smv-norm-k100, smv-no-norm-k100)
- Score Distribution Statistics (sigma-x0.5, sigma-max)
- Relative Score Deviation (RSD)

### Supervised QPP
- BERT-QPP (Cross-Encoder)
- BERT-QPP (Bi-Encoder)

## Evaluation Metrics

### Retrieval Metrics
- nDCG@5, nDCG@10
- MAP (Mean Average Precision)
- Other standard IR metrics

### Generation Metrics (Nuggetizer)
- **strict_vital_score**: Strict support for vital nuggets
- **strict_all_score**: Strict support for all nuggets
- **vital_score**: Support for vital nuggets (with partial credit)
- **all_score**: Support for all nuggets (with partial credit)

## Key Findings

1. **Utility Gap**: There is a systematic divergence between retrieval metrics (nDCG) and generation quality (nugget scores). Variants that maximize nDCG do not necessarily produce the best generated answers.

2. **QPP Effectiveness**: QPP can reliably identify variants that improve end-to-end answer quality over the original query, even when retrieval metrics suggest otherwise.

3. **Pre-retrieval Efficiency**: Lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG.

4. **Intra-topic Discrimination**: QPP methods can effectively discriminate among query variants of the same information need, not just across different topics.

## Experimental Setup

- **RAG Model**: GPT-4o
- **Top-k Documents**: 5 documents per query for generation
- **Retrieval Depth**: Top 100 documents retrieved, top 5 used for generation
- **Evaluation**: Nuggetizer framework with NIST nuggets

## Reproducibility

All code, data, and results are provided in this repository:
- Query variants: `querygym/queries/`
- RAG results: `querygym/rag_results/`
- Nuggetizer scores: `querygym/rag_nuggetized_eval/`
- QPP predictions: `querygym/qpp/` (including `bert_qpp_results/` and `QSDQPP/`)

See `REPRODUCIBILITY.md` for detailed instructions.

## Contact

For questions about the paper or code, please open an issue on GitHub.

## Acknowledgments

This work uses:
- TREC-RAG dataset from TREC 2024 RAG Track
- Ragnarok framework for RAG pipeline
- Nuggetizer framework for answer evaluation
- Pyserini for retrieval
- Various QPP implementations from QPP4CS and other sources
