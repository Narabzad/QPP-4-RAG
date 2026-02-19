#!/usr/bin/env python3
"""
Analyze correlations between nugget scores and QPP metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_consolidated_data(json_file):
    """Load the consolidated query data."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_data_for_correlation(data):
    """Extract nugget scores and QPP metrics for correlation analysis."""
    pyserini_data = []
    cohere_data = []
    
    for query_id, query_data in data.items():
        for reformulation in query_data['reformulations']:
            # Extract pyserini data
            nugget_pyserini = reformulation.get('nugget_scores', {}).get('retrieval', {})
            qpp_pyserini = reformulation.get('qpp_metrics', {}).get('pyserini', {})
            
            if nugget_pyserini and qpp_pyserini:
                row_pyserini = {
                    'query_id': query_id,
                    'method': reformulation.get('method'),
                    'trial': reformulation.get('trial'),
                    **nugget_pyserini,
                    **qpp_pyserini
                }
                pyserini_data.append(row_pyserini)
            
            # Extract cohere data
            nugget_cohere = reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
            qpp_cohere = reformulation.get('qpp_metrics', {}).get('cohere', {})
            
            if nugget_cohere and qpp_cohere:
                row_cohere = {
                    'query_id': query_id,
                    'method': reformulation.get('method'),
                    'trial': reformulation.get('trial'),
                    **nugget_cohere,
                    **qpp_cohere
                }
                cohere_data.append(row_cohere)
    
    return pd.DataFrame(pyserini_data), pd.DataFrame(cohere_data)

def calculate_correlations(df, nugget_cols, qpp_cols, method_name):
    """Calculate Pearson and Spearman correlations."""
    results = []
    
    for nugget_col in nugget_cols:
        if nugget_col not in df.columns:
            continue
            
        for qpp_col in qpp_cols:
            if qpp_col not in df.columns:
                continue
            
            # Remove NaN values
            valid_mask = ~(df[nugget_col].isna() | df[qpp_col].isna())
            if valid_mask.sum() < 3:  # Need at least 3 points for correlation
                continue
            
            x = df.loc[valid_mask, nugget_col]
            y = df.loc[valid_mask, qpp_col]
            
            # Calculate correlations
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            
            results.append({
                'nugget_metric': nugget_col,
                'qpp_metric': qpp_col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': valid_mask.sum()
            })
    
    return pd.DataFrame(results)

def create_correlation_heatmap(corr_df, method_name, output_dir):
    """Create correlation heatmap."""
    # Pivot for heatmap
    pearson_pivot = corr_df.pivot(index='nugget_metric', columns='qpp_metric', values='pearson_r')
    spearman_pivot = corr_df.pivot(index='nugget_metric', columns='qpp_metric', values='spearman_r')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # Pearson correlation
    sns.heatmap(pearson_pivot, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Pearson r'})
    axes[0].set_title(f'{method_name} - Pearson Correlation\n(Nugget Scores vs QPP Metrics)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('QPP Metrics', fontsize=12)
    axes[0].set_ylabel('Nugget Scores', fontsize=12)
    
    # Spearman correlation
    sns.heatmap(spearman_pivot, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Spearman ρ'})
    axes[1].set_title(f'{method_name} - Spearman Correlation\n(Nugget Scores vs QPP Metrics)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('QPP Metrics', fontsize=12)
    axes[1].set_ylabel('Nugget Scores', fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / f'{method_name.lower().replace(" ", "_")}_correlation_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved heatmap: {output_file}")
    
    return pearson_pivot, spearman_pivot

def create_scatter_plots(df, nugget_cols, qpp_cols, method_name, output_dir, top_n=5):
    """Create scatter plots for top correlations."""
    # Calculate all correlations first
    corr_results = []
    for nugget_col in nugget_cols:
        if nugget_col not in df.columns:
            continue
        for qpp_col in qpp_cols:
            if qpp_col not in df.columns:
                continue
            
            valid_mask = ~(df[nugget_col].isna() | df[qpp_col].isna())
            if valid_mask.sum() < 3:
                continue
            
            x = df.loc[valid_mask, nugget_col]
            y = df.loc[valid_mask, qpp_col]
            pearson_r, _ = pearsonr(x, y)
            
            corr_results.append({
                'nugget': nugget_col,
                'qpp': qpp_col,
                'correlation': abs(pearson_r),
                'pearson_r': pearson_r
            })
    
    corr_df = pd.DataFrame(corr_results)
    if len(corr_df) == 0:
        return
    
    # Get top correlations (by absolute value)
    top_corrs = corr_df.nlargest(top_n, 'correlation')
    
    n_plots = len(top_corrs)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, row in top_corrs.iterrows():
        ax = axes[top_corrs.index.get_loc(idx)]
        nugget = row['nugget']
        qpp = row['qpp']
        corr = row['pearson_r']
        
        valid_mask = ~(df[nugget].isna() | df[qpp].isna())
        x = df.loc[valid_mask, nugget]
        y = df.loc[valid_mask, qpp]
        
        ax.scatter(x, y, alpha=0.5, s=20)
        ax.set_xlabel(nugget, fontsize=10)
        ax.set_ylabel(qpp, fontsize=10)
        ax.set_title(f'r = {corr:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{method_name} - Top {n_plots} Correlations (Nugget Scores vs QPP Metrics)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = output_dir / f'{method_name.lower().replace(" ", "_")}_scatter_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved scatter plots: {output_file}")

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    json_file = base_dir / "consolidated_query_data.json"
    output_dir = base_dir / "correlation_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("📖 Loading consolidated data...")
    data = load_consolidated_data(json_file)
    print(f"✅ Loaded data for {len(data)} queries")
    
    print("\n🔄 Extracting data for correlation analysis...")
    df_pyserini, df_cohere = extract_data_for_correlation(data)
    print(f"✅ Pyserini: {len(df_pyserini)} samples")
    print(f"✅ Cohere: {len(df_cohere)} samples")
    
    # Define nugget and QPP metric columns
    nugget_cols = ['strict_vital_score', 'strict_all_score', 'vital_score', 'all_score']
    qpp_cols = [
        'clarity-score-k10', 'clarity-score-k100',
        'wig-norm-k100', 'wig-no-norm-k100', 'wig-norm-k1000', 'wig-no-norm-k1000',
        'nqc-norm-k100', 'nqc-no-norm-k100',
        'smv-norm-k100', 'smv-no-norm-k100',
        'sigma-x0.5', 'sigma-max', 'RSD'
    ]
    
    # Analyze Pyserini
    print("\n📊 Analyzing Pyserini correlations...")
    corr_pyserini = calculate_correlations(df_pyserini, nugget_cols, qpp_cols, "Pyserini")
    corr_pyserini = corr_pyserini.sort_values('pearson_r', key=abs, ascending=False)
    
    output_csv_pyserini = output_dir / "pyserini_correlations.csv"
    corr_pyserini.to_csv(output_csv_pyserini, index=False)
    print(f"  ✅ Saved correlations: {output_csv_pyserini}")
    
    # Create visualizations for Pyserini
    if len(corr_pyserini) > 0:
        create_correlation_heatmap(corr_pyserini, "Pyserini", output_dir)
        create_scatter_plots(df_pyserini, nugget_cols, qpp_cols, "Pyserini", output_dir)
    
    # Analyze Cohere
    print("\n📊 Analyzing Cohere correlations...")
    corr_cohere = calculate_correlations(df_cohere, nugget_cols, qpp_cols, "Cohere")
    corr_cohere = corr_cohere.sort_values('pearson_r', key=abs, ascending=False)
    
    output_csv_cohere = output_dir / "cohere_correlations.csv"
    corr_cohere.to_csv(output_csv_cohere, index=False)
    print(f"  ✅ Saved correlations: {output_csv_cohere}")
    
    # Create visualizations for Cohere
    if len(corr_cohere) > 0:
        create_correlation_heatmap(corr_cohere, "Cohere", output_dir)
        create_scatter_plots(df_cohere, nugget_cols, qpp_cols, "Cohere", output_dir)
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("\n" + "="*80)
    print("PY SERINI - Top 10 Correlations (by absolute Pearson r):")
    print("="*80)
    top_pyserini = corr_pyserini.head(10)[['nugget_metric', 'qpp_metric', 'pearson_r', 'spearman_r', 'n_samples']]
    print(top_pyserini.to_string(index=False))
    
    print("\n" + "="*80)
    print("COHERE - Top 10 Correlations (by absolute Pearson r):")
    print("="*80)
    top_cohere = corr_cohere.head(10)[['nugget_metric', 'qpp_metric', 'pearson_r', 'spearman_r', 'n_samples']]
    print(top_cohere.to_string(index=False))
    
    print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
