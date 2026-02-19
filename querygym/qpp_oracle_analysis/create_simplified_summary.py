#!/usr/bin/env python3
"""
Create simplified summary CSV files with just the mean performance metrics.
"""

import pandas as pd
from pathlib import Path

def create_simplified_summary():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym/qpp_oracle_analysis")
    
    # Read the detailed CSV files
    df_pyserini = pd.read_csv(base_dir / "qpp_oracle_performance_pyserini.csv")
    df_cohere = pd.read_csv(base_dir / "qpp_oracle_performance_cohere.csv")
    
    # Create simplified versions with just mean metrics
    cols_to_keep = [
        'qpp_metric',
        'retrieval_method',
        'nugget_strict_vital_score_mean',
        'nugget_strict_all_score_mean',
        'nugget_vital_score_mean',
        'nugget_all_score_mean',
        'retrieval_ndcg@10_mean',
        'retrieval_recall@100_mean'
    ]
    
    # Rename columns for clarity
    rename_dict = {
        'nugget_strict_vital_score_mean': 'nugget_strict_vital',
        'nugget_strict_all_score_mean': 'nugget_strict_all',
        'nugget_vital_score_mean': 'nugget_vital',
        'nugget_all_score_mean': 'nugget_all',
        'retrieval_ndcg@10_mean': 'ndcg@10',
        'retrieval_recall@100_mean': 'recall@100'
    }
    
    df_pyserini_simple = df_pyserini[cols_to_keep].copy()
    df_pyserini_simple.rename(columns=rename_dict, inplace=True)
    
    # Separate baseline/max rows from QPP metrics
    baseline_rows_pyserini = df_pyserini_simple[df_pyserini_simple['qpp_metric'].str.startswith(('original', 'max_oracle'))].copy()
    qpp_rows_pyserini = df_pyserini_simple[~df_pyserini_simple['qpp_metric'].str.startswith(('original', 'max_oracle'))].copy()
    qpp_rows_pyserini = qpp_rows_pyserini.sort_values('nugget_all', ascending=False)
    # Combine: baseline rows first, then sorted QPP metrics
    df_pyserini_simple = pd.concat([baseline_rows_pyserini, qpp_rows_pyserini], ignore_index=True)
    
    df_cohere_simple = df_cohere[cols_to_keep].copy()
    df_cohere_simple.rename(columns=rename_dict, inplace=True)
    
    # Separate baseline/max rows from QPP metrics
    baseline_rows_cohere = df_cohere_simple[df_cohere_simple['qpp_metric'].str.startswith(('original', 'max_oracle'))].copy()
    qpp_rows_cohere = df_cohere_simple[~df_cohere_simple['qpp_metric'].str.startswith(('original', 'max_oracle'))].copy()
    qpp_rows_cohere = qpp_rows_cohere.sort_values('nugget_all', ascending=False)
    # Combine: baseline rows first, then sorted QPP metrics
    df_cohere_simple = pd.concat([baseline_rows_cohere, qpp_rows_cohere], ignore_index=True)
    
    # Save simplified versions
    output_pyserini = base_dir / "qpp_oracle_performance_pyserini_simple.csv"
    output_cohere = base_dir / "qpp_oracle_performance_cohere_simple.csv"
    
    df_pyserini_simple.to_csv(output_pyserini, index=False, float_format='%.4f')
    df_cohere_simple.to_csv(output_cohere, index=False, float_format='%.4f')
    
    print(f"✅ Created simplified summary: {output_pyserini}")
    print(f"✅ Created simplified summary: {output_cohere}")
    
    # Print top 5 for each
    print("\n" + "="*80)
    print("PY SERINI - Top 5 QPP Metrics (by nugget_all):")
    print("="*80)
    print(df_pyserini_simple.head(5).to_string(index=False))
    
    print("\n" + "="*80)
    print("COHERE - Top 5 QPP Metrics (by nugget_all):")
    print("="*80)
    print(df_cohere_simple.head(5).to_string(index=False))

if __name__ == "__main__":
    create_simplified_summary()
