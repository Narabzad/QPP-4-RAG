#!/usr/bin/env python3
"""
Add generation-only nugget scores to consolidated_query_data.json.
"""

import json
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent


def load_generation_only_scores(scores_dir):
    """Load all generation-only nugget scores from score files."""
    scores_dir = Path(scores_dir)
    scores_by_method = {}
    
    score_files = sorted(scores_dir.glob("*_scores.jsonl"))
    print(f"📁 Found {len(score_files)} score files")
    
    for score_file in score_files:
        # Extract method and trial from filename
        # Format: rag_results_run.{method}_{trial}_gpt_4o_mini_top3_top0_scores.jsonl
        filename = score_file.stem.replace("_scores", "")
        
        # Remove prefix and suffix
        filename = filename.replace("rag_results_run.", "")
        filename = filename.replace("_gpt_4o_mini_top3_top0", "")
        
        # Parse method and trial
        if filename == "original":
            method = "original"
            trial = None
        else:
            # Split by last underscore to get trial
            parts = filename.rsplit("_", 1)
            if len(parts) == 2 and parts[1].startswith("trial"):
                method = parts[0]
                trial = int(parts[1].replace("trial", ""))
            else:
                method = filename
                trial = None
        
        # Load scores
        scores = {}
        with open(score_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                qid = data.get('qid')
                if qid and qid != 'all':
                    scores[qid] = {
                        'strict_vital_score': data.get('strict_vital_score'),
                        'strict_all_score': data.get('strict_all_score'),
                        'vital_score': data.get('vital_score'),
                        'all_score': data.get('all_score')
                    }
        
        key = (method, trial)
        scores_by_method[key] = scores
        print(f"  ✅ Loaded {method} trial {trial}: {len(scores)} queries")
    
    return scores_by_method


def add_generation_only_to_consolidated(consolidated_file, generation_only_scores, output_file):
    """Add generation-only scores to consolidated data."""
    
    # Load consolidated data
    print(f"\n📖 Loading consolidated data from {consolidated_file}...")
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} queries")
    
    # Track statistics
    total_updates = 0
    method_stats = defaultdict(lambda: {'queries_updated': 0, 'queries_not_found': 0})
    
    # Add generation-only scores to each reformulation
    for qid, query_data in data.items():
        for reformulation in query_data['reformulations']:
            method = reformulation['method']
            trial = reformulation['trial']
            
            key = (method, trial)
            
            if key in generation_only_scores:
                scores = generation_only_scores[key]
                if qid in scores:
                    # Add generation-only nugget scores
                    reformulation['generation_only_nugget_scores'] = scores[qid]
                    total_updates += 1
                    method_stats[key]['queries_updated'] += 1
                else:
                    method_stats[key]['queries_not_found'] += 1
    
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
    print(f"\nPer-method breakdown:")
    print("-" * 80)
    
    for (method, trial), stats in sorted(method_stats.items()):
        trial_str = f"trial{trial}" if trial else "None"
        print(f"{method:20s} {trial_str:10s} | Updated: {stats['queries_updated']:3d} | Not found: {stats['queries_not_found']:3d}")
    
    return data


def main():
    base_dir = _REPO / "querygym"
    
    consolidated_file = base_dir / "consolidated_query_data.json"
    scores_dir = base_dir / "rag_nuggetized_eval_o" / "scores"
    output_file = base_dir / "consolidated_query_data.json"  # Overwrite original
    
    print("="*80)
    print("Adding Generation-Only Scores to Consolidated Data")
    print("="*80)
    
    # Load generation-only scores
    generation_only_scores = load_generation_only_scores(scores_dir)
    
    if not generation_only_scores:
        print("❌ No generation-only scores found!")
        return
    
    # Add to consolidated data
    updated_data = add_generation_only_to_consolidated(
        consolidated_file,
        generation_only_scores,
        output_file
    )
    
    print("\n" + "="*80)
    print("✅ Generation-Only Scores Added Successfully!")
    print("="*80)
    print(f"📁 Updated file: {output_file}")


if __name__ == "__main__":
    main()
