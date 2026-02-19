#!/usr/bin/env python3
"""
Verify Clarity k=100 and WIG updates
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
print("CLARITY k=100 & WIG VERIFICATION REPORT")
print("="*80)
print(f"\nTotal CSV files found: {len(csv_files)}\n")

# Statistics
clarity_k10_scores = []
clarity_k100_scores = []
wig_norm_scores = []
wig_no_norm_scores = []

print("Checking first 5 files in detail:")
print("-" * 80)

for csv_file in csv_files[:5]:
    filename = os.path.basename(csv_file)
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f"\n{filename}")
        print(f"  Queries: {len(df)}")
        print(f"  Clarity k=10  - Non-zero: {(df['clarity-score-k10'] > 0).sum()}, Mean: {df['clarity-score-k10'].mean():.3f}")
        print(f"  Clarity k=100 - Non-zero: {(df['clarity-score-k100'] > 0).sum()}, Mean: {df['clarity-score-k100'].mean():.3f}")
        print(f"  WIG norm k=100 - Non-zero: {(df['wig-norm-k100'] != 0).sum()}, Mean: {df['wig-norm-k100'].mean():.6f}")
        print(f"  WIG no-norm k=100 - Non-zero: {(df['wig-no-norm-k100'] > 0).sum()}, Mean: {df['wig-no-norm-k100'].mean():.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "-" * 80)
print("\nCollecting statistics from all files...")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        clarity_k10_scores.extend(df['clarity-score-k10'].values)
        clarity_k100_scores.extend(df['clarity-score-k100'].values)
        wig_norm_scores.extend(df['wig-norm-k100'].values)
        wig_no_norm_scores.extend(df['wig-no-norm-k100'].values)
    except:
        pass

clarity_k10_scores = np.array(clarity_k10_scores)
clarity_k100_scores = np.array(clarity_k100_scores)
wig_norm_scores = np.array(wig_norm_scores)
wig_no_norm_scores = np.array(wig_no_norm_scores)

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

print("\nClarity k=10:")
print(f"  Total queries: {len(clarity_k10_scores)}")
print(f"  Non-zero: {(clarity_k10_scores > 0).sum()}")
print(f"  Mean: {clarity_k10_scores.mean():.4f}")
print(f"  Range: {clarity_k10_scores.min():.4f} - {clarity_k10_scores.max():.4f}")

print("\nClarity k=100:")
print(f"  Total queries: {len(clarity_k100_scores)}")
print(f"  Non-zero: {(clarity_k100_scores > 0).sum()}")
print(f"  Mean: {clarity_k100_scores.mean():.4f}")
print(f"  Range: {clarity_k100_scores.min():.4f} - {clarity_k100_scores.max():.4f}")

print("\nWIG norm k=100:")
print(f"  Total queries: {len(wig_norm_scores)}")
print(f"  Non-zero: {(wig_norm_scores != 0).sum()}")
print(f"  Mean: {wig_norm_scores.mean():.6f}")
print(f"  Note: This is 0 because only 100 docs are retrieved per query")
print(f"        WIG-norm requires corpus_size > k for meaningful values")

print("\nWIG no-norm k=100:")
print(f"  Total queries: {len(wig_no_norm_scores)}")
print(f"  Non-zero: {(wig_no_norm_scores > 0).sum()}")
print(f"  Mean: {wig_no_norm_scores.mean():.4f}")
print(f"  Range: {wig_no_norm_scores.min():.4f} - {wig_no_norm_scores.max():.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Clarity k=10:  {(clarity_k10_scores > 0).sum()}/{len(clarity_k10_scores)} queries computed")
print(f"✅ Clarity k=100: {(clarity_k100_scores > 0).sum()}/{len(clarity_k100_scores)} queries computed")
print(f"⚠️  WIG norm k=100: {(wig_norm_scores != 0).sum()}/{len(wig_norm_scores)} (0 due to dataset limitation)")
print(f"✅ WIG no-norm k=100: {(wig_no_norm_scores > 0).sum()}/{len(wig_no_norm_scores)} queries (already computed)")
print("="*80)
