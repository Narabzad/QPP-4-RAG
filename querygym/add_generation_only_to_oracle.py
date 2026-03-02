#!/usr/bin/env python3
"""
Add generation-only performance as a baseline to oracle analysis CSV files.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def calculate_generation_only_performance(consolidated_file):
    """Calculate average performance for generation-only across all reformulations."""
    
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all generation-only scores
    all_scores = {
        'strict_vital': [],
        'strict_all': [],
        'vital': [],
        'all': []
    }
    
    for qid, query_data in data.items():
        for reformulation in query_data['reformulations']:
            gen_scores = reformulation.get('generation_only_nugget_scores', {})
            if gen_scores:
                all_scores['strict_vital'].append(gen_scores.get('strict_vital_score', 0))
                all_scores['strict_all'].append(gen_scores.get('strict_all_score', 0))
                all_scores['vital'].append(gen_scores.get('vital_score', 0))
                all_scores['all'].append(gen_scores.get('all_score', 0))
    
    # Calculate means and stds
    result = {
        'qpp_metric': 'generation_only_baseline',
        'retrieval_method': 'none',
        'n_queries': len(data),
        'nugget_strict_vital_score_mean': np.mean(all_scores['strict_vital']) if all_scores['strict_vital'] else 0,
        'nugget_strict_vital_score_std': np.std(all_scores['strict_vital']) if all_scores['strict_vital'] else 0,
        'nugget_strict_vital_score_n': len(all_scores['strict_vital']),
        'nugget_strict_all_score_mean': np.mean(all_scores['strict_all']) if all_scores['strict_all'] else 0,
        'nugget_strict_all_score_std': np.std(all_scores['strict_all']) if all_scores['strict_all'] else 0,
        'nugget_strict_all_score_n': len(all_scores['strict_all']),
        'nugget_vital_score_mean': np.mean(all_scores['vital']) if all_scores['vital'] else 0,
        'nugget_vital_score_std': np.std(all_scores['vital']) if all_scores['vital'] else 0,
        'nugget_vital_score_n': len(all_scores['vital']),
        'nugget_all_score_mean': np.mean(all_scores['all']) if all_scores['all'] else 0,
        'nugget_all_score_std': np.std(all_scores['all']) if all_scores['all'] else 0,
        'nugget_all_score_n': len(all_scores['all']),
        # No retrieval metrics for generation-only
        'retrieval_ndcg@10_mean': None,
        'retrieval_ndcg@10_std': None,
        'retrieval_ndcg@10_n': 0,
        'retrieval_recall@100_mean': None,
        'retrieval_recall@100_std': None,
        'retrieval_recall@100_n': 0
    }
    
    return result


def add_generation_only_row(csv_file, generation_only_row):
    """Add generation-only baseline row to oracle CSV file."""
    
    # Load existing CSV
    df = pd.read_csv(csv_file)
    
    # Add generation-only row at the end
    new_row = pd.DataFrame([generation_only_row])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save back
    df.to_csv(csv_file, index=False)
    print(f"✅ Added generation-only baseline to {csv_file}")
    
    return df


def main():
    base_dir = _REPO / "querygym"
    
    consolidated_file = base_dir / "consolidated_query_data.json"
    oracle_dir = base_dir / "qpp_oracle_analysis"
    
    print("="*80)
    print("Adding Generation-Only Baseline to Oracle Analysis")
    print("="*80)
    
    # Calculate generation-only performance
    print("\n📊 Calculating generation-only performance...")
    gen_only_row = calculate_generation_only_performance(consolidated_file)
    
    print(f"✅ Generation-only baseline:")
    print(f"   Nugget all score: {gen_only_row['nugget_all_score_mean']:.4f}")
    print(f"   Nugget strict all score: {gen_only_row['nugget_strict_all_score_mean']:.4f}")
    print(f"   Total samples: {gen_only_row['nugget_all_score_n']}")
    
    # Add to oracle CSV files
    print("\n📁 Adding to oracle CSV files...")
    
    csv_files = [
        oracle_dir / "qpp_oracle_performance_pyserini.csv",
        oracle_dir / "qpp_oracle_performance_pyserini_simple.csv",
        oracle_dir / "qpp_oracle_performance_cohere.csv",
        oracle_dir / "qpp_oracle_performance_cohere_simple.csv",
    ]
    
    for csv_file in csv_files:
        if csv_file.exists():
            # Update retrieval_method for each file
            row = gen_only_row.copy()
            if 'pyserini' in csv_file.name:
                row['retrieval_method'] = 'pyserini'
            elif 'cohere' in csv_file.name:
                row['retrieval_method'] = 'cohere'
            
            add_generation_only_row(csv_file, row)
        else:
            print(f"⚠️  File not found: {csv_file}")
    
    print("\n" + "="*80)
    print("✅ Generation-Only Baseline Added to All Oracle Files!")
    print("="*80)


if __name__ == "__main__":
    main()
