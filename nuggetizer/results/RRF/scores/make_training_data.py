#!/usr/bin/env python3
"""
Script to create training data from RRF scores files.

This script:
1. Excludes queries that exist in the test qrels file
2. Extracts version numbers from filenames
3. Creates training data with qid, versions, and performance scores
"""

import json
import os
import re
import csv
from pathlib import Path

def load_test_qids(test_qrels_path):
    """Load QIDs from the test qrels file to exclude them."""
    test_qids = set()
    with open(test_qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                test_qids.add(data['qid'])
    return test_qids

def extract_versions_from_filename(filename):
    """Extract version numbers from filename.
    
    Example: rag_results_ragnarok_format_run.rag24.pyserini_v0_1_12_37_43_rep21_gpt_4o_mini_assignments.jsonl
    Returns: [0, 1, 12, 37, 43]
    """
    # Look for pattern v followed by numbers separated by underscores
    pattern = r'v(\d+(?:_\d+)*)'
    match = re.search(pattern, filename)
    if match:
        version_str = match.group(1)
        versions = [int(v) for v in version_str.split('_')]
        return versions
    return []

def process_scores_file(file_path, test_qids):
    """Process a single scores file and return training data."""
    training_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = data['qid']
                
                # Skip if this QID is in test set
                if qid in test_qids:
                    continue
                
                # Skip the "all" summary line
                if qid == "all":
                    continue
                
                # Extract versions from filename
                filename = os.path.basename(file_path)
                versions = extract_versions_from_filename(filename)
                
                if not versions:
                    print(f"Warning: Could not extract versions from {filename}")
                    continue
                
                # Get performance score (all_score)
                performance = data.get('all_score', 0.0)
                
                training_data.append({
                    'qid': qid,
                    'versions': versions,
                    'performance': performance
                })
    
    return training_data

def main():
    # Paths
    scores_dir = Path(__file__).parent
    test_qrels_path = Path("/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl")
    output_path = scores_dir / "training_data.csv"
    
    print(f"Processing scores from: {scores_dir}")
    print(f"Excluding QIDs from: {test_qrels_path}")
    print(f"Output file: {output_path}")
    
    # Load test QIDs to exclude
    test_qids = load_test_qids(test_qrels_path)
    print(f"Found {len(test_qids)} test QIDs to exclude")
    
    # Process all scores files
    all_training_data = []
    scores_files = list(scores_dir.glob("rag_results_ragnarok_format_run.rag24.pyserini_v*_gpt_4o_mini_assignments.jsonl"))
    
    print(f"Found {len(scores_files)} scores files to process")
    
    for file_path in scores_files:
        print(f"Processing: {file_path.name}")
        training_data = process_scores_file(file_path, test_qids)
        all_training_data.extend(training_data)
    
    print(f"Total training samples: {len(all_training_data)}")
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['qid', 'version_0', 'version_1', 'version_2', 'version_3', 'version_4', 'performance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in all_training_data:
            row = {
                'qid': data['qid'],
                'version_0': data['versions'][0] if len(data['versions']) > 0 else None,
                'version_1': data['versions'][1] if len(data['versions']) > 1 else None,
                'version_2': data['versions'][2] if len(data['versions']) > 2 else None,
                'version_3': data['versions'][3] if len(data['versions']) > 3 else None,
                'version_4': data['versions'][4] if len(data['versions']) > 4 else None,
                'performance': data['performance']
            }
            writer.writerow(row)
    
    print(f"Training data written to: {output_path}")
    
    # Print some statistics
    unique_qids = set(data['qid'] for data in all_training_data)
    print(f"Unique QIDs in training data: {len(unique_qids)}")
    
    # Show sample of the data
    print("\nSample of training data:")
    for i, data in enumerate(all_training_data[:5]):
        print(f"  {i+1}. QID: {data['qid']}, Versions: {data['versions']}, Performance: {data['performance']:.4f}")

if __name__ == "__main__":
    main()
