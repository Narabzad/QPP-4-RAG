#!/usr/bin/env python3
"""
Verify and summarize Clarity scores across all QPP metric files
"""

import pandas as pd
import glob
import os
import numpy as np

qpp_dir = "/future/u/negara/home/set_based_QPP/querygym/qpp"

# Find all post-retrieval CSV files
csv_files = glob.glob(os.path.join(qpp_dir, "post_retrieval_*_qpp_metrics.csv"))
csv_files.sort()

print("="*80)
print("CLARITY SCORE VERIFICATION REPORT")
print("="*80)
print(f"\nTotal CSV files found: {len(csv_files)}")
print()

# Statistics
total_files = 0
files_with_clarity = 0
total_queries = 0
queries_with_nonzero_clarity = 0
all_clarity_scores = []

print("File-by-File Summary:")
print("-" * 80)

for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    
    try:
        df = pd.read_csv(csv_file)
        total_files += 1
        
        if 'clarity-score-k10' in df.columns:
            clarity_scores = df['clarity-score-k10'].values
            nonzero_count = (clarity_scores > 0).sum()
            
            if nonzero_count > 0:
                files_with_clarity += 1
                total_queries += len(df)
                queries_with_nonzero_clarity += nonzero_count
                all_clarity_scores.extend(clarity_scores[clarity_scores > 0])
                
                mean_clarity = clarity_scores[clarity_scores > 0].mean()
                min_clarity = clarity_scores[clarity_scores > 0].min()
                max_clarity = clarity_scores[clarity_scores > 0].max()
                
                print(f"✓ {filename[:60]:<60} | {nonzero_count:>2} queries | avg: {mean_clarity:>6.3f}")
            else:
                print(f"⚠ {filename[:60]:<60} | No non-zero scores")
        else:
            print(f"✗ {filename[:60]:<60} | No clarity column")
            
    except Exception as e:
        print(f"✗ {filename[:60]:<60} | Error: {str(e)[:20]}")

print("-" * 80)
print()
print("="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Files processed:                {total_files}")
print(f"Files with Clarity scores:      {files_with_clarity}")
print(f"Total queries:                  {total_queries}")
print(f"Queries with non-zero Clarity:  {queries_with_nonzero_clarity}")
print()

if all_clarity_scores:
    all_clarity_scores = np.array(all_clarity_scores)
    print("Clarity Score Distribution:")
    print(f"  Mean:       {all_clarity_scores.mean():.4f}")
    print(f"  Median:     {np.median(all_clarity_scores):.4f}")
    print(f"  Std Dev:    {all_clarity_scores.std():.4f}")
    print(f"  Min:        {all_clarity_scores.min():.4f}")
    print(f"  Max:        {all_clarity_scores.max():.4f}")
    print(f"  25th %ile:  {np.percentile(all_clarity_scores, 25):.4f}")
    print(f"  75th %ile:  {np.percentile(all_clarity_scores, 75):.4f}")
    print()
    
    # Histogram
    print("Score Distribution (histogram):")
    bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    hist, _ = np.histogram(all_clarity_scores, bins=bins)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / max(hist) * 50)
        print(f"  {bins[i]:>4.1f}-{bins[i+1]:>4.1f}: {bar} ({hist[i]})")

print()
print("="*80)
print("✅ VERIFICATION COMPLETE")
print("="*80)
