#!/usr/bin/env python3
"""
Consolidate RRF (Reciprocal Rank Fusion) retrieval data:
- Nugget evaluation scores
- Retrieval evaluation metrics (NDCG@10, Recall@100)
into a single JSON file.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_original_queries(queries_file):
    """Load original queries from topics file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id] = query_text
    return queries

def load_nugget_scores(jsonl_file):
    """Load nugget evaluation scores from JSONL file."""
    scores = {}
    if not Path(jsonl_file).exists():
        return scores
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get('qid')
            if qid and qid != 'all':  # Skip aggregate 'all' entry
                scores[qid] = {
                    'strict_vital_score': data.get('strict_vital_score'),
                    'strict_all_score': data.get('strict_all_score'),
                    'vital_score': data.get('vital_score'),
                    'all_score': data.get('all_score')
                }
    return scores

def load_retrieval_metrics(jsonl_file):
    """Load retrieval evaluation metrics (NDCG@10, Recall@100) from JSONL file."""
    metrics = {}
    if not Path(jsonl_file).exists():
        return metrics
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get('qid')
            if qid and qid != 'all':  # Skip aggregate 'all' entry
                metrics[qid] = {
                    'ndcg@10': data.get('ndcg_cut_10'),
                    'recall@100': data.get('recall_100')
                }
    return metrics

def extract_rrf_metric_name(filename):
    """Extract RRF metric name from filename.
    Example: rag_results_run.rrf_IDF_avg_gpt_4o_top5_scores.jsonl -> rrf_IDF_avg
    """
    name = filename.stem  # Remove .jsonl
    # Remove prefix and suffix
    if name.startswith('rag_results_run.'):
        name = name[16:]  # Remove 'rag_results_run.'
    if name.endswith('_gpt_4o_top5_scores'):
        name = name[:-19]  # Remove '_gpt_4o_top5_scores' (19 chars)
    return name

def consolidate_rrf_data(
    queries_file,
    rrf_nugget_pyserini_dir,
    rrf_nugget_cohere_dir,
    rrf_eval_pyserini_dir,
    rrf_eval_cohere_dir,
    output_file
):
    """Consolidate all RRF data into a single JSON file."""
    
    print("📖 Loading original queries...")
    original_queries = load_original_queries(queries_file)
    print(f"✅ Loaded {len(original_queries)} queries")
    
    # Initialize consolidated data structure
    consolidated = {}
    for query_id, query_text in original_queries.items():
        consolidated[query_id] = {
            'query': query_text,
            'rrf_runs': []
        }
    
    # Find all RRF nugget files
    rrf_nugget_pyserini_path = Path(rrf_nugget_pyserini_dir)
    rrf_nugget_cohere_path = Path(rrf_nugget_cohere_dir)
    
    pyserini_nugget_files = sorted(rrf_nugget_pyserini_path.glob("*.jsonl"))
    cohere_nugget_files = sorted(rrf_nugget_cohere_path.glob("*.jsonl"))
    
    print(f"\n📁 Found {len(pyserini_nugget_files)} pyserini RRF nugget files")
    print(f"📁 Found {len(cohere_nugget_files)} cohere RRF nugget files")
    
    # Create mapping from metric name to files
    pyserini_nugget_map = {}
    for file in pyserini_nugget_files:
        metric_name = extract_rrf_metric_name(file)
        pyserini_nugget_map[metric_name] = file
    
    cohere_nugget_map = {}
    for file in cohere_nugget_files:
        metric_name = extract_rrf_metric_name(file)
        cohere_nugget_map[metric_name] = file
    
    # Find all RRF evaluation files
    rrf_eval_pyserini_path = Path(rrf_eval_pyserini_dir)
    rrf_eval_cohere_path = Path(rrf_eval_cohere_dir)
    
    pyserini_eval_files = sorted(rrf_eval_pyserini_path.glob("*.jsonl"))
    cohere_eval_files = sorted(rrf_eval_cohere_path.glob("*.jsonl"))
    
    print(f"📁 Found {len(pyserini_eval_files)} pyserini RRF evaluation files")
    print(f"📁 Found {len(cohere_eval_files)} cohere RRF evaluation files")
    
    # Create mapping from metric name to evaluation files
    pyserini_eval_map = {}
    for file in pyserini_eval_files:
        # Format: retrieval_RRF_pyserini_rrf_IDF_avg_per_query.jsonl -> rrf_IDF_avg
        name = file.stem
        if name.startswith('retrieval_RRF_pyserini_'):
            metric_name = name[23:]  # Remove 'retrieval_RRF_pyserini_'
        if metric_name.endswith('_per_query'):
            metric_name = metric_name[:-10]  # Remove '_per_query'
        pyserini_eval_map[metric_name] = file
    
    cohere_eval_map = {}
    for file in cohere_eval_files:
        # Format: retrieval_RRF_cohere_rrf_IDF_avg_per_query.jsonl -> rrf_IDF_avg
        name = file.stem
        if name.startswith('retrieval_RRF_cohere_'):
            metric_name = name[21:]  # Remove 'retrieval_RRF_cohere_'
        if metric_name.endswith('_per_query'):
            metric_name = metric_name[:-10]  # Remove '_per_query'
        cohere_eval_map[metric_name] = file
    
    # Get all unique RRF metric names
    all_metric_names = sorted(set(list(pyserini_nugget_map.keys()) + 
                                   list(cohere_nugget_map.keys()) +
                                   list(pyserini_eval_map.keys()) +
                                   list(cohere_eval_map.keys())))
    
    print(f"\n🔄 Processing {len(all_metric_names)} RRF metrics...")
    
    for metric_name in all_metric_names:
        print(f"  Processing {metric_name}...")
        
        # Load nugget scores
        nugget_scores_pyserini = {}
        nugget_scores_cohere = {}
        
        if metric_name in pyserini_nugget_map:
            nugget_scores_pyserini = load_nugget_scores(pyserini_nugget_map[metric_name])
        
        if metric_name in cohere_nugget_map:
            nugget_scores_cohere = load_nugget_scores(cohere_nugget_map[metric_name])
        
        # Load retrieval evaluation metrics
        retrieval_metrics_pyserini = {}
        retrieval_metrics_cohere = {}
        
        if metric_name in pyserini_eval_map:
            retrieval_metrics_pyserini = load_retrieval_metrics(pyserini_eval_map[metric_name])
        
        if metric_name in cohere_eval_map:
            retrieval_metrics_cohere = load_retrieval_metrics(cohere_eval_map[metric_name])
        
        # Add data for each query
        for query_id in original_queries.keys():
            rrf_run_data = {
                'metric_name': metric_name,
                'nugget_scores': {
                    'pyserini': nugget_scores_pyserini.get(query_id, {}),
                    'cohere': nugget_scores_cohere.get(query_id, {})
                },
                'retrieval_metrics': {
                    'pyserini': retrieval_metrics_pyserini.get(query_id, {}),
                    'cohere': retrieval_metrics_cohere.get(query_id, {})
                }
            }
            consolidated[query_id]['rrf_runs'].append(rrf_run_data)
    
    print(f"\n💾 Saving consolidated RRF data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(consolidated)} queries with RRF data")
    print(f"📊 Total RRF runs per query: {len(consolidated[list(consolidated.keys())[0]]['rrf_runs'])}")
    
    return consolidated

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    
    queries_file = base_dir / "queries" / "topics.original.txt"
    rrf_nugget_pyserini_dir = base_dir / "rag_nuggetized_eval_RRF" / "retrieval_RRF_pyserini" / "scores"
    rrf_nugget_cohere_dir = base_dir / "rag_nuggetized_eval_RRF" / "retrieval_RRF_cohere" / "scores"
    rrf_eval_pyserini_dir = base_dir / "retrieval_eval_RRF"
    rrf_eval_cohere_dir = base_dir / "retrieval_eval_RRF"
    output_file = base_dir / "consolidated_rrf_data.json"
    
    consolidate_rrf_data(
        queries_file,
        rrf_nugget_pyserini_dir,
        rrf_nugget_cohere_dir,
        rrf_eval_pyserini_dir,
        rrf_eval_cohere_dir,
        output_file
    )
    
    print(f"\n🎉 Done! Output saved to: {output_file}")

if __name__ == "__main__":
    main()
