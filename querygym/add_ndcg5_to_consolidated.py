#!/usr/bin/env python3
"""
Add ndcg_cut_5 (ndcg@5) to consolidated_query_data.json from retrieval_eval files.
"""

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def load_retrieval_metrics_with_ndcg5(jsonl_file):
    """Load retrieval metrics including ndcg@5 from JSONL file."""
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
            if qid:
                metrics[qid] = {
                    'ndcg@5': data.get('ndcg_cut_5'),
                    'ndcg@10': data.get('ndcg_cut_10'),
                    'recall@100': data.get('recall_100')
                }
    return metrics


def add_ndcg5_to_consolidated(consolidated_file, retrieval_eval_dir, output_file):
    """Add ndcg@5 to consolidated data."""
    
    # Load consolidated data
    print(f"📖 Loading consolidated data from {consolidated_file}...")
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} queries")
    
    # Track statistics
    total_updates = 0
    methods_updated = set()
    
    # Get all reformulation methods and trials
    methods = ['original', 'genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    
    print("\n🔄 Processing retrieval eval files...")
    
    for method in methods:
        if method == 'original':
            trials = [None]
        else:
            trials = [1, 2, 3, 4, 5]
        
        for trial in trials:
            # Construct filenames for both retrieval methods
            if trial is None:
                pyserini_file = retrieval_eval_dir / f"retrieval_{method}_per_query.jsonl"
                cohere_file = retrieval_eval_dir / f"retrieval_cohere_{method}_per_query.jsonl"
                method_key = method
                trial_key = None
            else:
                pyserini_file = retrieval_eval_dir / f"retrieval_{method}_trial{trial}_per_query.jsonl"
                cohere_file = retrieval_eval_dir / f"retrieval_cohere_{method}_trial{trial}_per_query.jsonl"
                method_key = method
                trial_key = trial
            
            # Load metrics
            pyserini_metrics = load_retrieval_metrics_with_ndcg5(pyserini_file)
            cohere_metrics = load_retrieval_metrics_with_ndcg5(cohere_file)
            
            if not pyserini_metrics and not cohere_metrics:
                continue
            
            print(f"  Processing {method} trial {trial}...")
            
            # Update consolidated data
            for qid, query_data in data.items():
                for reformulation in query_data['reformulations']:
                    if reformulation['method'] == method_key and reformulation['trial'] == trial_key:
                        # Add ndcg@5 to retrieval_metrics
                        if 'retrieval_metrics' not in reformulation:
                            reformulation['retrieval_metrics'] = {'pyserini': {}, 'cohere': {}}
                        
                        # Update pyserini metrics
                        if qid in pyserini_metrics and pyserini_metrics[qid].get('ndcg@5') is not None:
                            reformulation['retrieval_metrics']['pyserini']['ndcg@5'] = pyserini_metrics[qid]['ndcg@5']
                            total_updates += 1
                        
                        # Update cohere metrics
                        if qid in cohere_metrics and cohere_metrics[qid].get('ndcg@5') is not None:
                            reformulation['retrieval_metrics']['cohere']['ndcg@5'] = cohere_metrics[qid]['ndcg@5']
                            total_updates += 1
                        
                        methods_updated.add((method, trial))
                        break
    
    # Save updated consolidated data
    print(f"\n💾 Saving updated consolidated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {output_file}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nTotal updates: {total_updates}")
    print(f"Methods updated: {len(methods_updated)}")
    print(f"\nMethods processed:")
    for method, trial in sorted(methods_updated):
        trial_str = f"trial{trial}" if trial else "None"
        print(f"  {method:20s} {trial_str}")
    
    return data


def main():
    base_dir = _REPO / "querygym"
    
    consolidated_file = base_dir / "consolidated_query_data.json"
    retrieval_eval_dir = base_dir / "retrieval_eval"
    output_file = base_dir / "consolidated_query_data.json"  # Overwrite
    
    print("="*80)
    print("Adding ndcg@5 to Consolidated Data")
    print("="*80)
    
    # Add ndcg@5
    updated_data = add_ndcg5_to_consolidated(
        consolidated_file,
        retrieval_eval_dir,
        output_file
    )
    
    print("\n" + "="*80)
    print("✅ ndcg@5 Added Successfully!")
    print("="*80)
    print(f"📁 Updated file: {output_file}")


if __name__ == "__main__":
    main()
