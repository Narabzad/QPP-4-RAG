#!/usr/bin/env python3
"""
Create filtered oracle CSV with only key columns, combining pyserini and cohere.
"""

import pandas as pd
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def create_filtered_oracle(pyserini_file, cohere_file, output_file):
    """Create filtered oracle CSV with key columns only."""
    
    # Load both files
    df_pys = pd.read_csv(pyserini_file)
    df_coh = pd.read_csv(cohere_file)
    
    # Combine
    df_combined = pd.concat([df_pys, df_coh], ignore_index=True)
    
    # Select only specified columns
    columns_to_keep = [
        'qpp_metric',
        'retrieval_method',
        'nugget_strict_vital_score_mean',
        'nugget_all_score_mean',
        'genonly_strict_vital_score_mean',
        'genonly_all_score_mean',
        'retrieval_ndcg@5_mean',
        'retrieval_recall@100_mean'
    ]
    
    df_filtered = df_combined[columns_to_keep].copy()
    
    # Save
    df_filtered.to_csv(output_file, index=False)
    
    return df_filtered


def main():
    base_dir = _REPO / "querygym"
    oracle_dir = base_dir / "qpp_oracle_analysis"
    final_dir = oracle_dir / "final"
    final_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Creating Filtered Oracle CSV")
    print("="*80)
    
    # Create filtered version
    pyserini_file = oracle_dir / "qpp_oracle_performance_pyserini.csv"
    cohere_file = oracle_dir / "qpp_oracle_performance_cohere.csv"
    output_file = final_dir / "qpp_oracle_performance_filtered.csv"
    
    print(f"\n📖 Loading oracle files...")
    print(f"  - {pyserini_file.name}")
    print(f"  - {cohere_file.name}")
    
    df_filtered = create_filtered_oracle(pyserini_file, cohere_file, output_file)
    
    print(f"\n✅ Created filtered oracle CSV:")
    print(f"  📁 File: {output_file}")
    print(f"  📊 Rows: {len(df_filtered)}")
    print(f"  📊 Columns: {len(df_filtered.columns)}")
    
    print(f"\n📋 Columns included:")
    for i, col in enumerate(df_filtered.columns, 1):
        print(f"  {i}. {col}")
    
    # Show sample data
    print(f"\n📊 Sample Data (top 10 by nugget_all_score_mean):")
    print("="*80)
    top10 = df_filtered.nlargest(10, 'nugget_all_score_mean')
    display_cols = ['qpp_metric', 'retrieval_method', 'nugget_all_score_mean', 'genonly_all_score_mean']
    print(top10[display_cols].to_string(index=False))
    
    print(f"\n✅ Filtered oracle file saved to: {output_file}")


if __name__ == "__main__":
    main()
