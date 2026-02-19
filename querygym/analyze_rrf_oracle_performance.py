#!/usr/bin/env python3
"""
Analyze RRF oracle performance: For each RRF run (based on different QPP metrics),
report the aggregated nugget scores and retrieval metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_consolidated_rrf_data(json_file):
    """Load the consolidated RRF data."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_rrf_performance(data, retrieval_method='pyserini'):
    """
    For each RRF run (metric), aggregate the performance metrics.
    
    Args:
        data: Consolidated RRF data
        retrieval_method: 'pyserini' or 'cohere'
    
    Returns:
        DataFrame with one row per RRF metric showing aggregated performance
    """
    results = []
    
    # Get all unique RRF metric names
    all_metrics = set()
    for query_id, query_data in data.items():
        for rrf_run in query_data.get('rrf_runs', []):
            metric_name = rrf_run.get('metric_name')
            if metric_name:
                all_metrics.add(metric_name)
    
    all_metrics = sorted(list(all_metrics))
    print(f"📊 Analyzing {len(all_metrics)} RRF metrics for {retrieval_method}...")
    
    for metric_name in all_metrics:
        print(f"  Processing {metric_name}...")
        
        selected_performances = {
            'nugget_strict_vital_score': [],
            'nugget_strict_all_score': [],
            'nugget_vital_score': [],
            'nugget_all_score': [],
            'retrieval_ndcg@10': [],
            'retrieval_recall@100': []
        }
        
        queries_processed = 0
        
        for query_id, query_data in data.items():
            # Find RRF run with this metric name
            for rrf_run in query_data.get('rrf_runs', []):
                if rrf_run.get('metric_name') == metric_name:
                    queries_processed += 1
                    
                    # Get nugget scores
                    nugget_scores = rrf_run.get('nugget_scores', {}).get(retrieval_method, {})
                    
                    # Get retrieval metrics
                    retrieval_metrics = rrf_run.get('retrieval_metrics', {}).get(retrieval_method, {})
                    
                    # Collect metrics
                    if nugget_scores:
                        selected_performances['nugget_strict_vital_score'].append(nugget_scores.get('strict_vital_score'))
                        selected_performances['nugget_strict_all_score'].append(nugget_scores.get('strict_all_score'))
                        selected_performances['nugget_vital_score'].append(nugget_scores.get('vital_score'))
                        selected_performances['nugget_all_score'].append(nugget_scores.get('all_score'))
                    
                    if retrieval_metrics:
                        selected_performances['retrieval_ndcg@10'].append(retrieval_metrics.get('ndcg@10'))
                        selected_performances['retrieval_recall@100'].append(retrieval_metrics.get('recall@100'))
                    
                    break
        
        # Calculate averages (ignoring None/NaN values)
        result_row = {
            'rrf_metric': metric_name,
            'retrieval_method': retrieval_method,
            'n_queries': queries_processed
        }
        
        for metric_name_inner, values in selected_performances.items():
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            if valid_values:
                result_row[f'{metric_name_inner}_mean'] = np.mean(valid_values)
                result_row[f'{metric_name_inner}_std'] = np.std(valid_values)
                result_row[f'{metric_name_inner}_n'] = len(valid_values)
            else:
                result_row[f'{metric_name_inner}_mean'] = None
                result_row[f'{metric_name_inner}_std'] = None
                result_row[f'{metric_name_inner}_n'] = 0
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_all_score'):
    """
    Calculate maximum oracle performance: for each query, select the RRF run
    with the highest actual performance (not based on QPP).
    
    Args:
        performance_metric: Which metric to use for selecting best RRF run
                           ('nugget_all_score', 'nugget_strict_all_score', 'retrieval_ndcg@10', etc.)
    """
    selected_performances = {
        'nugget_strict_vital_score': [],
        'nugget_strict_all_score': [],
        'nugget_vital_score': [],
        'nugget_all_score': [],
        'retrieval_ndcg@10': [],
        'retrieval_recall@100': []
    }
    
    queries_processed = 0
    
    for query_id, query_data in data.items():
        best_rrf_run = None
        best_performance_value = None
        
        # Find RRF run with highest performance for the specified metric
        for rrf_run in query_data.get('rrf_runs', []):
            # Get nugget scores
            nugget_scores = rrf_run.get('nugget_scores', {}).get(retrieval_method, {})
            
            # Get retrieval metrics
            retrieval_metrics = rrf_run.get('retrieval_metrics', {}).get(retrieval_method, {})
            
            # Determine which metric to use for selection
            if performance_metric.startswith('nugget_'):
                metric_value = nugget_scores.get(performance_metric.replace('nugget_', ''))
            elif performance_metric.startswith('retrieval_'):
                metric_value = retrieval_metrics.get(performance_metric.replace('retrieval_', ''))
            else:
                metric_value = None
            
            if metric_value is not None and not np.isnan(metric_value):
                if best_performance_value is None or metric_value > best_performance_value:
                    best_performance_value = metric_value
                    best_rrf_run = rrf_run
        
        # Collect metrics from best RRF run
        if best_rrf_run is not None:
            queries_processed += 1
            
            # Get nugget scores
            nugget_scores = best_rrf_run.get('nugget_scores', {}).get(retrieval_method, {})
            
            # Get retrieval metrics
            retrieval_metrics = best_rrf_run.get('retrieval_metrics', {}).get(retrieval_method, {})
            
            # Collect all metrics
            if nugget_scores:
                selected_performances['nugget_strict_vital_score'].append(nugget_scores.get('strict_vital_score'))
                selected_performances['nugget_strict_all_score'].append(nugget_scores.get('strict_all_score'))
                selected_performances['nugget_vital_score'].append(nugget_scores.get('vital_score'))
                selected_performances['nugget_all_score'].append(nugget_scores.get('all_score'))
            
            if retrieval_metrics:
                selected_performances['retrieval_ndcg@10'].append(retrieval_metrics.get('ndcg@10'))
                selected_performances['retrieval_recall@100'].append(retrieval_metrics.get('recall@100'))
    
    # Calculate averages
    result_row = {
        'rrf_metric': 'max_oracle',
        'retrieval_method': retrieval_method,
        'n_queries': queries_processed
    }
    
    for metric_name, values in selected_performances.items():
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            result_row[f'{metric_name}_mean'] = np.mean(valid_values)
            result_row[f'{metric_name}_std'] = np.std(valid_values)
            result_row[f'{metric_name}_n'] = len(valid_values)
        else:
            result_row[f'{metric_name}_mean'] = None
            result_row[f'{metric_name}_std'] = None
            result_row[f'{metric_name}_n'] = 0
    
    return result_row

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    json_file = base_dir / "consolidated_rrf_data.json"
    output_dir = base_dir / "qpp_oracle_analysis_RRF"
    output_dir.mkdir(exist_ok=True)
    
    print("📖 Loading consolidated RRF data...")
    data = load_consolidated_rrf_data(json_file)
    print(f"✅ Loaded data for {len(data)} queries")
    
    # Analyze for both retrieval methods
    print("\n" + "="*80)
    print("ANALYZING RRF PYSERINI RETRIEVAL")
    print("="*80)
    df_pyserini = analyze_rrf_performance(data, retrieval_method='pyserini')
    
    # Add max oracle rows
    print("\n  Adding maximum oracle performance (based on nugget_all_score)...")
    max_oracle_row_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_all_score')
    max_oracle_row_pyserini['rrf_metric'] = 'max_oracle_nugget_all'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_row_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@10)...")
    max_oracle_ndcg_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='retrieval_ndcg@10')
    max_oracle_ndcg_pyserini['rrf_metric'] = 'max_oracle_ndcg@10'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_ndcg_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_recall@100)...")
    max_oracle_recall_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='retrieval_recall@100')
    max_oracle_recall_pyserini['rrf_metric'] = 'max_oracle_recall@100'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_recall_pyserini])], ignore_index=True)
    
    output_csv_pyserini = output_dir / "rrf_oracle_performance_pyserini.csv"
    df_pyserini.to_csv(output_csv_pyserini, index=False)
    print(f"\n✅ Saved Pyserini RRF results: {output_csv_pyserini}")
    
    # Create simplified version
    simple_columns = [
        'rrf_metric', 'retrieval_method', 'n_queries',
        'nugget_strict_vital_score_mean',
        'nugget_strict_all_score_mean',
        'nugget_vital_score_mean',
        'nugget_all_score_mean',
        'retrieval_ndcg@10_mean',
        'retrieval_recall@100_mean'
    ]
    df_pyserini_simple = df_pyserini[simple_columns].copy()
    # Round to 4 decimals
    for col in simple_columns[3:]:
        if col in df_pyserini_simple.columns:
            df_pyserini_simple[col] = df_pyserini_simple[col].round(4)
    output_csv_pyserini_simple = output_dir / "rrf_oracle_performance_pyserini_simple.csv"
    df_pyserini_simple.to_csv(output_csv_pyserini_simple, index=False)
    print(f"✅ Saved Pyserini RRF simplified results: {output_csv_pyserini_simple}")
    
    print("\n" + "="*80)
    print("ANALYZING RRF COHERE RETRIEVAL")
    print("="*80)
    df_cohere = analyze_rrf_performance(data, retrieval_method='cohere')
    
    # Add max oracle rows
    print("\n  Adding maximum oracle performance (based on nugget_all_score)...")
    max_oracle_row_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_all_score')
    max_oracle_row_cohere['rrf_metric'] = 'max_oracle_nugget_all'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_row_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@10)...")
    max_oracle_ndcg_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='retrieval_ndcg@10')
    max_oracle_ndcg_cohere['rrf_metric'] = 'max_oracle_ndcg@10'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_ndcg_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_recall@100)...")
    max_oracle_recall_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='retrieval_recall@100')
    max_oracle_recall_cohere['rrf_metric'] = 'max_oracle_recall@100'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_recall_cohere])], ignore_index=True)
    
    output_csv_cohere = output_dir / "rrf_oracle_performance_cohere.csv"
    df_cohere.to_csv(output_csv_cohere, index=False)
    print(f"\n✅ Saved Cohere RRF results: {output_csv_cohere}")
    
    # Create simplified version
    df_cohere_simple = df_cohere[simple_columns].copy()
    # Round to 4 decimals
    for col in simple_columns[3:]:
        if col in df_cohere_simple.columns:
            df_cohere_simple[col] = df_cohere_simple[col].round(4)
    output_csv_cohere_simple = output_dir / "rrf_oracle_performance_cohere_simple.csv"
    df_cohere_simple.to_csv(output_csv_cohere_simple, index=False)
    print(f"✅ Saved Cohere RRF simplified results: {output_csv_cohere_simple}")
    
    # Create combined file
    print("\n📊 Creating combined file...")
    df_combined = pd.concat([
        df_pyserini_simple,
        df_cohere_simple
    ], ignore_index=True)
    
    output_csv_combined = output_dir / "rrf_oracle_performance_combined.csv"
    df_combined.to_csv(output_csv_combined, index=False)
    print(f"✅ Saved combined RRF results: {output_csv_combined}")
    
    # Summary
    print("\n📊 Summary Statistics:")
    print("\n" + "="*80)
    print("PYSERINI RRF - Top 10 Metrics by Nugget All Score:")
    print("="*80)
    top_pyserini = df_pyserini_simple.nlargest(10, 'nugget_all_score_mean')[
        ['rrf_metric', 'nugget_all_score_mean', 'nugget_strict_all_score_mean', 
         'retrieval_ndcg@10_mean', 'retrieval_recall@100_mean']
    ]
    print(top_pyserini.to_string(index=False))
    
    print("\n" + "="*80)
    print("COHERE RRF - Top 10 Metrics by Nugget All Score:")
    print("="*80)
    top_cohere = df_cohere_simple.nlargest(10, 'nugget_all_score_mean')[
        ['rrf_metric', 'nugget_all_score_mean', 'nugget_strict_all_score_mean',
         'retrieval_ndcg@10_mean', 'retrieval_recall@100_mean']
    ]
    print(top_cohere.to_string(index=False))
    
    print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
