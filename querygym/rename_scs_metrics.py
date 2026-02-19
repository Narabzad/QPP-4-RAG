#!/usr/bin/env python3
"""
Rename SCS-1 to SCS-apx and SCS-2 to SCS-full in consolidated_query_data.json
"""

import json
from pathlib import Path

def rename_scs_metrics(data):
    """Rename SCS-1 to SCS-apx and SCS-2 to SCS-full"""
    renamed_count = 0
    
    for query_id, query_data in data.items():
        for reformulation in query_data.get('reformulations', []):
            # Check pre_retrieval_qpp_metrics
            if 'pre_retrieval_qpp_metrics' in reformulation:
                pre_metrics = reformulation['pre_retrieval_qpp_metrics']
                
                # Rename SCS-1 to SCS-apx
                if 'SCS-1' in pre_metrics:
                    pre_metrics['SCS-apx'] = pre_metrics.pop('SCS-1')
                    renamed_count += 1
                
                # Rename SCS-2 to SCS-full
                if 'SCS-2' in pre_metrics:
                    pre_metrics['SCS-full'] = pre_metrics.pop('SCS-2')
                    renamed_count += 1
    
    return renamed_count

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    json_file = base_dir / "consolidated_query_data.json"
    
    print("="*80)
    print("RENAMING SCS METRICS")
    print("="*80)
    print("  SCS-1 → SCS-apx (approximation)")
    print("  SCS-2 → SCS-full (full version)")
    print("="*80)
    
    print(f"\n📖 Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded data for {len(data)} queries")
    
    print("\n🔄 Renaming SCS metrics...")
    renamed_count = rename_scs_metrics(data)
    
    print(f"\n✅ Renamed {renamed_count} SCS metric entries")
    
    # Create backup
    backup_file = json_file.with_suffix('.json.backup2')
    print(f"\n💾 Creating backup: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Save updated file
    print(f"💾 Saving updated file: {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Done! Renamed SCS metrics:")
    print(f"   • SCS-1 → SCS-apx: {renamed_count // 2} entries")
    print(f"   • SCS-2 → SCS-full: {renamed_count // 2} entries")
    print(f"   Backup saved to: {backup_file}")

if __name__ == "__main__":
    main()
