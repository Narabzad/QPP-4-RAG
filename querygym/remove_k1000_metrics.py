#!/usr/bin/env python3
"""
Remove k1000 metrics from post-retrieval QPP metrics in consolidated_query_data.json
"""

import json
from pathlib import Path

def remove_k1000_metrics(data):
    """Remove all k1000 metrics from post-retrieval QPP metrics"""
    removed_count = 0
    
    for query_id, query_data in data.items():
        for reformulation in query_data.get('reformulations', []):
            # Check pyserini QPP metrics
            if 'qpp_metrics' in reformulation:
                pyserini_metrics = reformulation['qpp_metrics'].get('pyserini', {})
                cohere_metrics = reformulation['qpp_metrics'].get('cohere', {})
                
                # Remove k1000 metrics from pyserini
                keys_to_remove = [k for k in pyserini_metrics.keys() if 'k1000' in k]
                for key in keys_to_remove:
                    del pyserini_metrics[key]
                    removed_count += 1
                    print(f"  Removed {key} from pyserini metrics")
                
                # Remove k1000 metrics from cohere
                keys_to_remove = [k for k in cohere_metrics.keys() if 'k1000' in k]
                for key in keys_to_remove:
                    del cohere_metrics[key]
                    removed_count += 1
                    print(f"  Removed {key} from cohere metrics")
    
    return removed_count

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    json_file = base_dir / "consolidated_query_data.json"
    
    print("="*80)
    print("REMOVING k1000 METRICS FROM POST-RETRIEVAL QPP METRICS")
    print("="*80)
    
    print(f"\n📖 Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded data for {len(data)} queries")
    
    print("\n🗑️  Removing k1000 metrics...")
    removed_count = remove_k1000_metrics(data)
    
    print(f"\n✅ Removed {removed_count} k1000 metric entries")
    
    # Create backup
    backup_file = json_file.with_suffix('.json.backup')
    print(f"\n💾 Creating backup: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Save updated file
    print(f"💾 Saving updated file: {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Done! Removed {removed_count} k1000 metrics from post-retrieval QPP metrics")
    print(f"   Backup saved to: {backup_file}")

if __name__ == "__main__":
    main()
