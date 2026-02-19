#!/usr/bin/env python3
"""
Create combined oracle performance plots with ndcg@5, larger fonts, and better labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def create_oracle_plot(df, retrieval_method, ax):
    """Create oracle performance plot for one retrieval method."""
    
    # Filter data - exclude null and max_oracle rows for cleaner visualization
    df_plot = df[~df['qpp_metric'].str.contains('null|max_oracle', na=False)].copy()
    
    # Sort by nugget_all_score_mean
    df_plot = df_plot.sort_values('nugget_all_score_mean', ascending=True)
    
    # Create y-axis positions
    y_pos = np.arange(len(df_plot))
    
    # Plot bars
    width = 0.35
    
    # Retrieval performance
    bars1 = ax.barh(y_pos - width/2, df_plot['nugget_all_score_mean'], 
                    width, label='Retrieval (nugget_all_score)', 
                    color='steelblue', alpha=0.8)
    
    # Generation-only performance
    bars2 = ax.barh(y_pos + width/2, df_plot['genonly_all_score_mean'], 
                    width, label='Generation-Only (nugget_all_score)', 
                    color='coral', alpha=0.8)
    
    # Add NDCG@5 as scatter points
    ax2 = ax.twiny()
    scatter = ax2.scatter(df_plot['retrieval_ndcg@5_mean'], y_pos, 
                         marker='D', s=80, color='green', alpha=0.7,
                         label='NDCG@5', zorder=5, edgecolors='darkgreen', linewidths=1.5)
    
    # Set labels with larger fonts
    title_map = {
        'pyserini': 'Sparse Retriever (BM25)',
        'cohere': 'Dense Retriever (Cohere)'
    }
    ax.set_title(title_map.get(retrieval_method, retrieval_method), 
                fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('Nugget All Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('QPP Method', fontsize=14, fontweight='bold')
    ax2.set_xlabel('NDCG@5', fontsize=14, fontweight='bold', color='green')
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['qpp_metric'], fontsize=11)
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 0.6)
    ax2.set_xlim(0, 0.8)
    
    # Tick parameters
    ax.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12, colors='green')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legends
    legend1 = ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    legend2 = ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add value labels on bars for top 5
    top5_indices = y_pos[-5:]
    for i, idx in enumerate(top5_indices):
        ret_val = df_plot.iloc[idx]['nugget_all_score_mean']
        gen_val = df_plot.iloc[idx]['genonly_all_score_mean']
        
        # Retrieval bar label
        ax.text(ret_val + 0.01, idx - width/2, f'{ret_val:.3f}', 
               va='center', fontsize=10, fontweight='bold')
        
        # Generation-only bar label
        ax.text(gen_val + 0.01, idx + width/2, f'{gen_val:.3f}', 
               va='center', fontsize=10, fontweight='bold')
    
    return ax


def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    oracle_dir = base_dir / "qpp_oracle_analysis"
    
    # Load data
    df_pyserini = pd.read_csv(oracle_dir / "qpp_oracle_performance_pyserini.csv")
    df_cohere = pd.read_csv(oracle_dir / "qpp_oracle_performance_cohere.csv")
    
    print("📊 Creating combined oracle performance plots...")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
    
    # Create plots
    create_oracle_plot(df_pyserini, 'pyserini', ax1)
    create_oracle_plot(df_cohere, 'cohere', ax2)
    
    # Add overall title
    fig.suptitle('QPP Oracle Performance: Retrieval vs Generation-Only', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save as PNG
    output_png = oracle_dir / "qpp_oracle_performance_combined.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved PNG: {output_png}")
    
    # Save as PDF
    output_pdf = oracle_dir / "qpp_oracle_performance_combined.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Saved PDF: {output_pdf}")
    
    plt.close()
    
    # Also create separate PDFs and combine
    print("\n📄 Creating multi-page PDF with detailed views...")
    
    output_multipage = oracle_dir / "qpp_oracle_performance_detailed.pdf"
    with PdfPages(output_multipage) as pdf:
        # Page 1: Combined view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
        create_oracle_plot(df_pyserini, 'pyserini', ax1)
        create_oracle_plot(df_cohere, 'cohere', ax2)
        fig.suptitle('QPP Oracle Performance: Retrieval vs Generation-Only', 
                    fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Pyserini only (larger)
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        create_oracle_plot(df_pyserini, 'pyserini', ax)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Cohere only (larger)
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        create_oracle_plot(df_cohere, 'cohere', ax)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved multi-page PDF: {output_multipage}")
    
    print("\n" + "="*80)
    print("🎉 Plots created successfully!")
    print("="*80)
    print(f"📁 Output files:")
    print(f"   - {output_png.name} (high-res PNG)")
    print(f"   - {output_pdf.name} (single-page PDF)")
    print(f"   - {output_multipage.name} (3-page PDF with details)")


if __name__ == "__main__":
    main()
