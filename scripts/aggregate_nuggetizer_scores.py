#!/usr/bin/env python3
"""
Script to aggregate nuggetizer scores across all prediction folders.
Collects scores from all folders and calculates averages for the 4 metrics.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import statistics

_REPO = Path(__file__).resolve().parent.parent

def collect_scores_from_folder(folder_path):
    """Collect all scores from a single folder."""
    scores_dir = folder_path / "nuggetizer" / "scores"
    
    if not scores_dir.exists():
        print(f"⚠️  Scores directory not found: {scores_dir}")
        return []
    
    scores = []
    score_files = list(scores_dir.glob("*.jsonl"))
    
    for score_file in score_files:
        try:
            with open(score_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if data.get('qid') == 'all':
                            scores.append(data)
                            break  # Only take the first 'all' entry per file
        except Exception as e:
            print(f"❌ Error reading {score_file}: {e}")
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='Aggregate nuggetizer scores across all prediction folders')
    parser.add_argument('--predictions-dir', type=str,
                       default=str(_REPO / "predictions"),
                       help='Base predictions directory')
    parser.add_argument('--output-csv', type=str,
                       default=str(_REPO / "nuggetizer_scores_summary.csv"),
                       help='Output CSV file path')
    parser.add_argument('--folders', nargs='*', default=None,
                       help='Process only these specific folder names (default: all folders)')
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    # Get all folders to process
    all_folders = [f for f in predictions_dir.iterdir() if f.is_dir()]
    all_folders.sort()
    
    # Filter folders if specified
    if args.folders:
        all_folders = [f for f in all_folders if f.name in args.folders]
        print(f"🔍 Processing only specified folders: {args.folders}")
    
    print(f"📊 Found {len(all_folders)} folders to process")
    
    # Collect scores from all folders
    all_scores = []
    folder_stats = {}
    
    for folder in all_folders:
        print(f"📁 Processing folder: {folder.name}")
        scores = collect_scores_from_folder(folder)
        
        if scores:
            # Calculate averages for this folder
            folder_metrics = {
                'strict_vital_score': [],
                'strict_all_score': [],
                'vital_score': [],
                'all_score': []
            }
            
            for score in scores:
                for metric in folder_metrics:
                    if metric in score:
                        folder_metrics[metric].append(score[metric])
            
            # Calculate folder averages
            folder_averages = {}
            for metric, values in folder_metrics.items():
                if values:
                    folder_averages[metric] = statistics.mean(values)
                else:
                    folder_averages[metric] = 0.0
            
            folder_stats[folder.name] = folder_averages
            all_scores.extend(scores)
            print(f"   ✅ Found {len(scores)} score entries")
        else:
            print(f"   ⚠️  No scores found")
    
    if not all_scores:
        print("❌ No scores found in any folder")
        return
    
    # Calculate overall averages
    overall_metrics = {
        'strict_vital_score': [],
        'strict_all_score': [],
        'vital_score': [],
        'all_score': []
    }
    
    for score in all_scores:
        for metric in overall_metrics:
            if metric in score:
                overall_metrics[metric].append(score[metric])
    
    overall_averages = {}
    for metric, values in overall_metrics.items():
        if values:
            overall_averages[metric] = statistics.mean(values)
        else:
            overall_averages[metric] = 0.0
    
    # Write results to CSV
    output_file = Path(args.output_csv)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['folder', 'strict_vital_score', 'strict_all_score', 'vital_score', 'all_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Write folder-level averages
        for folder_name, metrics in folder_stats.items():
            row = {'folder': folder_name}
            row.update(metrics)
            writer.writerow(row)
        
        # Write overall average
        overall_row = {'folder': 'OVERALL_AVERAGE'}
        overall_row.update(overall_averages)
        writer.writerow(overall_row)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 NUGGETIZER SCORES SUMMARY")
    print(f"{'='*60}")
    
    print(f"📁 Processed {len(folder_stats)} folders")
    print(f"📊 Total score entries: {len(all_scores)}")
    print(f"💾 Results saved to: {output_file}")
    
    print(f"\n📈 Overall Averages:")
    for metric, avg in overall_averages.items():
        print(f"   {metric}: {avg:.4f}")
    
    print(f"\n📋 Folder-level averages:")
    for folder_name, metrics in folder_stats.items():
        print(f"   {folder_name}:")
        for metric, avg in metrics.items():
            print(f"      {metric}: {avg:.4f}")
    
    print(f"\n✅ Summary complete! Check {output_file} for detailed results.")

if __name__ == "__main__":
    main()



