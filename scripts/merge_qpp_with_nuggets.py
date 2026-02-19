#!/usr/bin/env python3
"""
Merge nugget scores with QPP metrics files.
"""
import csv
import json
import os
from pathlib import Path
from tqdm import tqdm

def load_nugget_scores(scores_file):
    """Load nugget scores from JSONL file."""
    scores = {}
    with open(scores_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('qid') != 'all':  # Skip global metrics
                scores[data['qid']] = {
                    'strict_vital_score': data['strict_vital_score'],
                    'strict_all_score': data['strict_all_score'],
                    'vital_score': data['vital_score'],
                    'all_score': data['all_score']
                }
    return scores

def format_number(value, decimal_places=4):
    """Format a number to specified decimal places, handling N/A values."""
    if value == 'N/A' or value is None or value == '':
        return 'N/A'
    try:
        num = float(value)
        return f"{num:.{decimal_places}f}"
    except (ValueError, TypeError):
        return 'N/A'

def merge_qpp_with_nuggets(qpp_file, nugget_scores, output_file):
    """Merge QPP metrics with nugget scores."""
    with open(qpp_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        
        # Filter out N/A columns from original fieldnames
        original_fieldnames = [col for col in reader.fieldnames if not col.startswith('N/A')]
        
        # Create new fieldnames including nugget scores
        fieldnames = original_fieldnames + [
            'strict_vital_score',
            'strict_all_score', 
            'vital_score',
            'all_score'
        ]
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            query_id = row['query_id']
            
            # Create new row with only non-N/A columns
            new_row = {col: row[col] for col in original_fieldnames}
            
            # Format numeric columns to 4 decimal places
            for col in new_row:
                if col != 'query_id':  # Don't format query_id
                    new_row[col] = format_number(new_row[col])
            
            # Add nugget scores if available
            if query_id in nugget_scores:
                scores = nugget_scores[query_id]
                new_row.update({
                    'strict_vital_score': format_number(scores['strict_vital_score']),
                    'strict_all_score': format_number(scores['strict_all_score']),
                    'vital_score': format_number(scores['vital_score']),
                    'all_score': format_number(scores['all_score'])
                })
            else:
                # Add N/A for missing scores
                new_row.update({
                    'strict_vital_score': 'N/A',
                    'strict_all_score': 'N/A',
                    'vital_score': 'N/A',
                    'all_score': 'N/A'
                })
            
            writer.writerow(new_row)

def find_matching_score_file(qpp_file):
    """Find the corresponding nugget score file for a QPP file."""
    qpp_path = Path(qpp_file)
    qpp_name = qpp_path.name
    
    # Extract run identifier from QPP filename
    # Examples:
    # post_retrieval_v0_run.rag24.cohere.v0_qpp_metrics.csv -> rag_results_ragnarok_format_run.rag24.cohere.v0_gpt_4o_mini.jsonl
    # pre_retrieval_v0_qpp_metrics_fast.csv -> rag_results_ragnarok_format_run.rag24.cohere.v0_gpt_4o_mini.jsonl
    
    if 'post_retrieval' in qpp_name:
        # Extract run info: post_retrieval_v0_run.rag24.cohere.v0_qpp_metrics.csv
        # Remove prefix and suffix
        clean_name = qpp_name.replace('post_retrieval_', '').replace('_qpp_metrics.csv', '')
        # Split by underscore to get parts: ['v0', 'run.rag24.cohere.v0']
        parts = clean_name.split('_')
        if len(parts) >= 2:
            # Take everything after the version: 'run.rag24.cohere.v0'
            run_info = '_'.join(parts[1:])
            score_filename = f"rag_results_ragnarok_format_{run_info}_gpt_4o_mini.jsonl"
        else:
            return None
    elif 'pre_retrieval' in qpp_name:
        # Extract version: pre_retrieval_v0_qpp_metrics_fast.csv
        version = qpp_name.replace('pre_retrieval_', '').replace('_qpp_metrics_fast.csv', '')
        # For pre-retrieval, we'll try to match with cohere files first
        score_filename = f"rag_results_ragnarok_format_run.rag24.cohere.{version}_gpt_4o_mini.jsonl"
    else:
        return None
    
    scores_dir = Path("/future/u/negara/home/set_based_QPP/nuggetizer/results/scores")
    score_file = scores_dir / score_filename
    
    if score_file.exists():
        return score_file
    
    # If cohere not found, try pyserini
    if 'cohere' in score_filename:
        score_filename = score_filename.replace('cohere', 'pyserini')
        score_file = scores_dir / score_filename
        if score_file.exists():
            return score_file
    
    return None

def main():
    # Define paths
    qpp_results_dir = Path("/future/u/negara/home/set_based_QPP/qpp_results")
    output_dir = Path("/future/u/negara/home/set_based_QPP/qpp_results_with_nuggets")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all QPP files
    qpp_files = list(qpp_results_dir.glob("*.csv"))
    print(f"Found {len(qpp_files)} QPP files to process")
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each QPP file
    for qpp_file in tqdm(qpp_files, desc="Processing QPP files"):
        # Find matching score file
        score_file = find_matching_score_file(qpp_file)
        
        if score_file is None:
            print(f"⚠️  No matching score file found for {qpp_file.name}")
            skipped += 1
            continue
        
        try:
            # Load nugget scores
            nugget_scores = load_nugget_scores(score_file)
            
            # Create output filename
            output_file = output_dir / f"{qpp_file.stem}_with_nuggets.csv"
            
            # Merge files
            merge_qpp_with_nuggets(qpp_file, nugget_scores, output_file)
            
            print(f"✅ Merged {qpp_file.name} with {score_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed to merge {qpp_file.name}: {e}")
            failed += 1
    
    # Summary
    print(f"\n🎉 Merging complete!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Skipped: {skipped}")
    print(f"   📁 Merged files saved to: {output_dir}")

if __name__ == "__main__":
    main()
