#!/usr/bin/env python3
"""
Create scatter plots for oracle performance analysis with ndcg@5.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def create_scatter_plot(df, retrieval_method, ax, show_legend=True):
    """Create scatter plot showing NDCG@5 vs nugget all score."""
    
    # Filter out null rows and generation-only oracle rows
    df_plot = df[df['qpp_metric'] != 'pre_null'].copy()
    df_plot = df_plot[~df_plot['qpp_metric'].str.contains('genonly', na=False)]
    
    # Separate different row types
    qpp_methods = df_plot[~df_plot['qpp_metric'].str.contains('max_oracle|avg_|original', na=False)]
    
    # Separate pre-retrieval and post-retrieval QPP methods
    pre_retrieval = qpp_methods[qpp_methods['qpp_metric'].str.startswith('pre_')]
    post_retrieval = qpp_methods[~qpp_methods['qpp_metric'].str.startswith('pre_')]
    
    # Filter oracle rows to only include the 4 specified ones
    allowed_oracles = [
        'max_oracle_ndcg@5',
        'max_oracle_nugget_all',
        'max_oracle_recall@100',
        'max_oracle_strict_vital_score'
    ]
    max_oracle_ret = df_plot[df_plot['qpp_metric'].isin(allowed_oracles)]
    
    avg_methods = df_plot[df_plot['qpp_metric'].str.contains('avg_', na=False)]
    original = df_plot[df_plot['qpp_metric'] == 'original']
    
    # Plot pre-retrieval QPP methods (less transparent for better visibility)
    if not pre_retrieval.empty:
        ax.scatter(pre_retrieval['retrieval_ndcg@5_mean'], 
                  pre_retrieval['nugget_all_score_mean'],
                  s=140, alpha=0.65, c='steelblue', marker='o',
                  label='Pre-Retrieval QPP', edgecolors='navy', linewidths=1.8)
    
    # Plot post-retrieval QPP methods (yellow, less transparent)
    if not post_retrieval.empty:
        ax.scatter(post_retrieval['retrieval_ndcg@5_mean'], 
                  post_retrieval['nugget_all_score_mean'],
                  s=140, alpha=0.65, c='gold', marker='s',
                  label='Post-Retrieval QPP', edgecolors='darkorange', linewidths=1.8)
    
    # Plot max oracle (only the 4 specified ones)
    if not max_oracle_ret.empty:
        ax.scatter(max_oracle_ret['retrieval_ndcg@5_mean'], 
                  max_oracle_ret['nugget_all_score_mean'],
                  s=250, alpha=0.75, c='red', marker='*',
                  label='Max Oracle', edgecolors='darkred', linewidths=2.5)
    
    # Plot query reformulation (avg methods) (less transparent)
    if not avg_methods.empty:
        ax.scatter(avg_methods['retrieval_ndcg@5_mean'], 
                  avg_methods['nugget_all_score_mean'],
                  s=180, alpha=0.7, c='green', marker='D',
                  label='Query Reformulation', edgecolors='darkgreen', linewidths=2)
    
    # Plot original (less transparent)
    if not original.empty:
        ax.scatter(original['retrieval_ndcg@5_mean'], 
                  original['nugget_all_score_mean'],
                  s=200, alpha=0.75, c='purple', marker='^',
                  label='Original', edgecolors='indigo', linewidths=2.5)
    
    # Labels and title
    title_map = {
        'pyserini': 'Sparse Retriever (BM25)',
        'cohere': 'Dense Retriever (Cohere)'
    }
    ax.set_title(title_map.get(retrieval_method, retrieval_method), 
                fontsize=32, fontweight='bold', pad=25)
    
    ax.set_xlabel('NDCG@5', fontsize=28, fontweight='bold')
    ax.set_ylabel('Nugget All Score', fontsize=28, fontweight='bold')
    
    # Set limits
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0.25, 0.55)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Legend with larger font (only if show_legend is True)
    if show_legend:
        ax.legend(fontsize=22, loc='upper left', framealpha=0.95, edgecolor='black', 
                 borderpad=1.2, labelspacing=1.2)
    
    # Tick parameters with larger font
    ax.tick_params(axis='both', labelsize=26)
    
    return ax


def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    oracle_dir = base_dir / "qpp_oracle_analysis"
    
    # Load data
    print("📖 Loading oracle performance data...")
    df_pyserini = pd.read_csv(oracle_dir / "qpp_oracle_performance_pyserini.csv")
    df_cohere = pd.read_csv(oracle_dir / "qpp_oracle_performance_cohere.csv")
    
    print(f"✅ Loaded Pyserini: {len(df_pyserini)} rows")
    print(f"✅ Loaded Cohere: {len(df_cohere)} rows")
    
    print("\n📊 Creating scatter plots...")
    
    # Create figure with two subplots side by side (ONLY THIS ONE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    
    # Create plots (left plot without legend, right plot with legend)
    create_scatter_plot(df_pyserini, 'pyserini', ax1, show_legend=False)
    create_scatter_plot(df_cohere, 'cohere', ax2, show_legend=True)
    
    # Adjust layout (no overall title)
    plt.tight_layout()
    
    # Save as PNG
    output_png = oracle_dir / "qpp_oracle_performance_scatter_combined.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved PNG: {output_png}")
    
    # Save as PDF (single page with both plots)
    output_pdf = oracle_dir / "qpp_oracle_performance_scatter_combined.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Saved PDF: {output_pdf}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("🎉 Scatter plot created successfully!")
    print("="*80)
    print(f"📁 Output files:")
    print(f"   - qpp_oracle_performance_scatter_combined.png (both plots)")
    print(f"   - qpp_oracle_performance_scatter_combined.pdf (both plots, side-by-side)")


if __name__ == "__main__":
    main()
