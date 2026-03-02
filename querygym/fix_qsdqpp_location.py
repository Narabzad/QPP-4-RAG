#!/usr/bin/env python3
"""
Move QSDQPP scores from qpp_metrics to pre_retrieval_qpp_metrics
"""

import json
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

CONSOLIDATED_FILE = str(_REPO / "querygym/consolidated_query_data.json")

def main():
    print("="*80)
    print("MOVING QSDQPP TO PRE-RETRIEVAL SECTION")
    print("="*80)
    
    # Load consolidated data
    print("\n📂 Loading consolidated_query_data.json...")
    with open(CONSOLIDATED_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded data for {len(data)} queries")
    
    # Move QSDQPP scores
    total_moved = 0
    total_reformulations = 0
    
    print("\n🔄 Moving QSDQPP scores to pre_retrieval_qpp_metrics...")
    
    for qid in tqdm(data.keys(), desc="Processing queries"):
        for reformulation in data[qid]['reformulations']:
            total_reformulations += 1
            
            # Check if QSDQPP is in qpp_metrics
            if 'qpp_metrics' in reformulation and 'qsdqpp_predicted_ndcg' in reformulation['qpp_metrics']:
                qsdqpp_score = reformulation['qpp_metrics']['qsdqpp_predicted_ndcg']
                
                # Remove from qpp_metrics
                del reformulation['qpp_metrics']['qsdqpp_predicted_ndcg']
                
                # Add to pre_retrieval_qpp_metrics
                if 'pre_retrieval_qpp_metrics' not in reformulation:
                    reformulation['pre_retrieval_qpp_metrics'] = {}
                
                reformulation['pre_retrieval_qpp_metrics']['qsdqpp_predicted_ndcg'] = qsdqpp_score
                
                total_moved += 1
    
    # Save updated consolidated data
    print("\n💾 Saving updated consolidated_query_data.json...")
    with open(CONSOLIDATED_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {CONSOLIDATED_FILE}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total reformulations: {total_reformulations}")
    print(f"QSDQPP scores moved: {total_moved}")
    print(f"Success rate: {total_moved/total_reformulations*100:.1f}%")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nQSDQPP scores moved to: pre_retrieval_qpp_metrics.qsdqpp_predicted_ndcg")

if __name__ == "__main__":
    main()
