#!/usr/bin/env python3
"""
Add QSDQPP predicted nDCG scores to consolidated_query_data.json
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

# Paths
QSDQPP_DIR = str(_REPO / "querygym/QSDQPP")
CONSOLIDATED_FILE = str(_REPO / "querygym/consolidated_query_data.json")
OUTPUT_FILE = str(_REPO / "querygym/consolidated_query_data.json")

def load_qsdqpp_scores(qsdqpp_file):
    """Load QSDQPP predicted nDCG scores from a file."""
    scores = {}
    
    with open(qsdqpp_file, 'r') as f:
        lines = f.readlines()
        
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    predicted_ndcg = float(parts[1])
                    scores[qid] = predicted_ndcg
    
    return scores

def main():
    print("="*80)
    print("ADDING QSDQPP SCORES TO CONSOLIDATED FILE")
    print("="*80)
    
    # Load consolidated data
    print("\n📂 Loading consolidated_query_data.json...")
    with open(CONSOLIDATED_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded data for {len(data)} queries")
    
    # Find all QSDQPP predicted_ndcg files
    qsdqpp_files = list(Path(QSDQPP_DIR).glob("topics.*_predicted_ndcg.txt"))
    print(f"\n📊 Found {len(qsdqpp_files)} QSDQPP files")
    
    # Process each QSDQPP file
    total_updates = 0
    method_stats = {}
    
    for qsdqpp_file in tqdm(qsdqpp_files, desc="Processing QSDQPP files"):
        # Extract method name from filename
        # Format: topics.{method}_predicted_ndcg.txt
        filename = qsdqpp_file.name
        method_name = filename.replace('topics.', '').replace('_predicted_ndcg.txt', '')
        
        # Load QSDQPP scores
        qsdqpp_scores = load_qsdqpp_scores(qsdqpp_file)
        
        method_stats[method_name] = {
            'queries_in_file': len(qsdqpp_scores),
            'queries_updated': 0,
            'queries_not_found': 0
        }
        
        # Add scores to consolidated data
        for qid in data.keys():
            # Find the reformulation for this method
            for reformulation in data[qid]['reformulations']:
                # Match method and trial
                ref_method = reformulation['method']
                ref_trial = reformulation['trial']
                
                # Construct the full method name
                if ref_trial:
                    full_method = f"{ref_method}_trial{ref_trial}"
                else:
                    full_method = ref_method
                
                if full_method == method_name:
                    # Add QSDQPP score
                    if qid in qsdqpp_scores:
                        # Add QSDQPP as a pre-retrieval metric
                        if 'pre_retrieval_qpp_metrics' not in reformulation:
                            reformulation['pre_retrieval_qpp_metrics'] = {}
                        
                        reformulation['pre_retrieval_qpp_metrics']['qsdqpp_predicted_ndcg'] = qsdqpp_scores[qid]
                        
                        method_stats[method_name]['queries_updated'] += 1
                        total_updates += 1
                    else:
                        method_stats[method_name]['queries_not_found'] += 1
                    
                    break
    
    # Save updated consolidated data
    print("\n💾 Saving updated consolidated_query_data.json...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {OUTPUT_FILE}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nTotal updates: {total_updates}")
    print(f"\nPer-method breakdown:")
    print("-" * 80)
    
    for method, stats in sorted(method_stats.items()):
        print(f"{method:30s}: {stats['queries_updated']:3d} updated, "
              f"{stats['queries_not_found']:3d} not found, "
              f"{stats['queries_in_file']:3d} in QSDQPP file")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nQSDQPP scores added to: {OUTPUT_FILE}")
    print(f"Field name: pre_retrieval_qpp_metrics.qsdqpp_predicted_ndcg")

if __name__ == "__main__":
    main()
