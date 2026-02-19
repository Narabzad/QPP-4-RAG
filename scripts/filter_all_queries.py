#!/usr/bin/env python3
"""
Filter all query variation files to only include the 56 queries that have qrels.
"""

import os
import json
from pathlib import Path

def load_qrels_queries():
    """Load the list of query IDs that have qrels."""
    qrels_file = "/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl"
    qrels_qids = set()
    
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                qrels_qids.add(data['qid'])
    
    print(f"📋 Loaded {len(qrels_qids)} queries with qrels")
    return qrels_qids

def filter_query_file(input_file, output_file, qrels_qids):
    """Filter a query file to only include queries with qrels."""
    filtered_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                qid = line.split('\t')[0]
                if qid in qrels_qids:
                    filtered_lines.append(line)
    
    # Write filtered file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    print(f"📝 Filtered {len(filtered_lines)}/{len(qrels_qids)} queries to {output_file.name}")
    return len(filtered_lines)

def main():
    print("🔍 Filtering All Query Variations")
    print("=" * 50)
    
    # Setup directories
    queries_dir = Path("/future/u/negara/home/set_based_QPP/data/query_reformulation/queries")
    output_dir = Path("/future/u/negara/home/set_based_QPP/multi_variation_results/temp_filtered_queries")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries with qrels
    qrels_qids = load_qrels_queries()
    
    # Find all variation files
    variation_files = list(queries_dir.glob("topics.rag24.test.v*.txt"))
    variation_files.sort(key=lambda x: int(x.stem.split('.')[-1][1:]))  # Sort by version number
    
    print(f"📁 Found {len(variation_files)} variation files")
    print(f"📂 Output directory: {output_dir}")
    
    # Process each variation
    successful_filters = 0
    failed_filters = 0
    
    for i, variation_file in enumerate(variation_files, 1):
        print(f"\n🔄 Processing variation {i}/{len(variation_files)}: {variation_file.name}")
        
        # Create output filename
        output_file = output_dir / f"{variation_file.stem}_filtered.txt"
        
        try:
            num_filtered = filter_query_file(variation_file, output_file, qrels_qids)
            if num_filtered > 0:
                successful_filters += 1
                print(f"✅ Successfully filtered {num_filtered} queries")
            else:
                failed_filters += 1
                print(f"⚠️  No queries found in {variation_file.name}")
        except Exception as e:
            failed_filters += 1
            print(f"❌ Error filtering {variation_file.name}: {e}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("🎉 FILTERING COMPLETED!")
    print(f"{'='*50}")
    
    print(f"✅ Successfully filtered: {successful_filters}")
    print(f"❌ Failed filters: {failed_filters}")
    print(f"📁 Total files processed: {len(variation_files)}")
    
    print(f"\n📂 Filtered files saved to: {output_dir}")
    print(f"💡 Each file contains only the 56 queries that have qrels")

if __name__ == "__main__":
    main()

