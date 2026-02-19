#!/usr/bin/env python3
"""
Create RRF retrieval results from top 5 QPP reformulations.

For each query and QPP method:
1. Get top 5 reformulations from top_qpp_reformulations.json
2. Load retrieval results for those reformulations
3. Perform RRF on the retrieval results
4. Save to appropriate output directory

Pre-retrieval QPP methods: use BOTH pyserini and cohere retrieval
Post-retrieval QPP methods: use pyserini if method starts with "pyserini_", cohere if "cohere_"
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_top_qpp_reformulations(json_file):
    """Load top QPP reformulations from JSON file."""
    print(f"📖 Loading top QPP reformulations from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded data for {len(data)} queries")
    return data


def get_retrieval_filename(method, trial):
    """Get retrieval filename for a given method and trial."""
    if method == 'original':
        return 'run.original.txt'
    else:
        return f'run.{method}_trial{trial}.txt'


def load_retrieval_file(retrieval_dir, filename):
    """Load a single retrieval file and return results as dict: qid -> list of (doc_id, rank, score)."""
    filepath = Path(retrieval_dir) / filename
    if not filepath.exists():
        return None
    
    results = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                qid = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                results[qid].append((doc_id, rank, score))
    
    return results


def perform_rrf(retrieval_lists, k=60):
    """
    Perform Reciprocal Rank Fusion on multiple retrieval lists.
    
    Args:
        retrieval_lists: List of retrieval result lists, each containing (doc_id, rank, score) tuples
        k: RRF constant (default: 60)
    
    Returns:
        List of (doc_id, rrf_score) tuples sorted by RRF score (descending)
    """
    doc_scores = defaultdict(float)
    
    for retrieval_list in retrieval_lists:
        for rank, (doc_id, _, _) in enumerate(retrieval_list, 1):
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score (descending)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_docs


def is_pre_retrieval_qpp(qpp_method):
    """Check if QPP method is pre-retrieval."""
    return not qpp_method.startswith('pyserini_') and not qpp_method.startswith('cohere_')


def get_retrieval_methods_for_qpp(qpp_method):
    """
    Determine which retrieval methods to use for a QPP method.
    
    Returns:
        List of retrieval method names: ['pyserini'], ['cohere'], or ['pyserini', 'cohere']
    """
    if is_pre_retrieval_qpp(qpp_method):
        # Pre-retrieval: use both
        return ['pyserini', 'cohere']
    elif qpp_method.startswith('pyserini_'):
        return ['pyserini']
    elif qpp_method.startswith('cohere_'):
        return ['cohere']
    else:
        # Default to both if unclear
        return ['pyserini', 'cohere']


def create_rrf_results(
    top_qpp_data,
    retrieval_pyserini_dir,
    retrieval_cohere_dir,
    output_pyserini_dir,
    output_cohere_dir,
    k=60
):
    """
    Create RRF results for all queries and QPP methods.
    
    Args:
        top_qpp_data: Dictionary from top_qpp_reformulations.json
        retrieval_pyserini_dir: Directory containing pyserini retrieval results
        retrieval_cohere_dir: Directory containing cohere retrieval results
        output_pyserini_dir: Output directory for pyserini RRF results
        output_cohere_dir: Output directory for cohere RRF results
        k: RRF constant
    """
    # Create output directories
    Path(output_pyserini_dir).mkdir(parents=True, exist_ok=True)
    Path(output_cohere_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all QPP methods
    all_qpp_methods = set()
    for qpp_methods_data in top_qpp_data.values():
        all_qpp_methods.update(qpp_methods_data.keys())
    
    # Track statistics
    stats = {
        'queries_processed': set(),
        'qpp_methods_processed': 0,
        'pyserini_rrf_files': 0,
        'cohere_rrf_files': 0
    }
    
    # Process each QPP method
    for qpp_method in tqdm(sorted(all_qpp_methods), desc="Processing QPP methods"):
        # Determine which retrieval methods to use
        retrieval_methods = get_retrieval_methods_for_qpp(qpp_method)
        
        # Process each retrieval method
        for retrieval_method in retrieval_methods:
            # Get appropriate directories
            if retrieval_method == 'pyserini':
                retrieval_dir = retrieval_pyserini_dir
                output_dir = output_pyserini_dir
            else:  # cohere
                retrieval_dir = retrieval_cohere_dir
                output_dir = output_cohere_dir
            
            # Collect RRF results for all queries for this QPP method
            all_rrf_results = []
            
            # Process each query
            for qid, qpp_methods_data in top_qpp_data.items():
                if qpp_method not in qpp_methods_data:
                    continue
                
                top_reformulations = qpp_methods_data[qpp_method]
                if not top_reformulations:
                    continue
                
                stats['queries_processed'].add(qid)
                
                # Collect retrieval results for top 5 reformulations
                retrieval_lists = []
                
                for reformulation in top_reformulations:
                    method = reformulation['method']
                    trial = reformulation['trial']
                    
                    # Get retrieval filename
                    filename = get_retrieval_filename(method, trial)
                    
                    # Load retrieval results for this reformulation
                    retrieval_results = load_retrieval_file(retrieval_dir, filename)
                    
                    if retrieval_results and qid in retrieval_results:
                        retrieval_lists.append(retrieval_results[qid])
                
                # Perform RRF if we have at least one retrieval list
                if retrieval_lists:
                    rrf_scores = perform_rrf(retrieval_lists, k=k)
                    
                    # Add to results for this QPP method
                    for rank, (doc_id, rrf_score) in enumerate(rrf_scores, 1):
                        all_rrf_results.append(
                            f"{qid} Q0 {doc_id} {rank} {rrf_score:.6f} rrf_{qpp_method}"
                        )
            
            # Write all RRF results for this QPP method to a single file
            if all_rrf_results:
                # Clean QPP method name for filename
                clean_qpp_name = qpp_method.replace('-', '_').replace(' ', '_').replace('/', '_')
                output_filename = f"run.rrf_{clean_qpp_name}.txt"
                output_filepath = Path(output_dir) / output_filename
                
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    for line in all_rrf_results:
                        f.write(line + '\n')
                
                stats['qpp_methods_processed'] += 1
                if retrieval_method == 'pyserini':
                    stats['pyserini_rrf_files'] += 1
                else:
                    stats['cohere_rrf_files'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Create RRF retrieval results from top 5 QPP reformulations'
    )
    parser.add_argument(
        '--top-qpp-file',
        type=str,
        default='top_qpp_reformulations.json',
        help='Path to top QPP reformulations JSON file'
    )
    parser.add_argument(
        '--retrieval-pyserini',
        type=str,
        default='retrieval',
        help='Directory containing pyserini retrieval results'
    )
    parser.add_argument(
        '--retrieval-cohere',
        type=str,
        default='retrieval_cohere',
        help='Directory containing cohere retrieval results'
    )
    parser.add_argument(
        '--output-pyserini',
        type=str,
        default='retrieval_RRF_pyserini',
        help='Output directory for pyserini RRF results'
    )
    parser.add_argument(
        '--output-cohere',
        type=str,
        default='retrieval_RRF_cohere',
        help='Output directory for cohere RRF results'
    )
    parser.add_argument(
        '--rrf-k',
        type=int,
        default=60,
        help='RRF constant k (default: 60)'
    )
    
    args = parser.parse_args()
    
    print("🎯 Starting RRF Results Creation from Top QPP Reformulations")
    print("=" * 70)
    
    # Load top QPP reformulations
    top_qpp_data = load_top_qpp_reformulations(args.top_qpp_file)
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent
    retrieval_pyserini_dir = base_dir / args.retrieval_pyserini
    retrieval_cohere_dir = base_dir / args.retrieval_cohere
    output_pyserini_dir = base_dir / args.output_pyserini
    output_cohere_dir = base_dir / args.output_cohere
    
    # Verify retrieval directories exist
    if not retrieval_pyserini_dir.exists():
        print(f"❌ Error: Pyserini retrieval directory not found: {retrieval_pyserini_dir}")
        return
    
    if not retrieval_cohere_dir.exists():
        print(f"❌ Error: Cohere retrieval directory not found: {retrieval_cohere_dir}")
        return
    
    print(f"\n📁 Pyserini retrieval directory: {retrieval_pyserini_dir}")
    print(f"📁 Cohere retrieval directory: {retrieval_cohere_dir}")
    print(f"📁 Pyserini RRF output directory: {output_pyserini_dir}")
    print(f"📁 Cohere RRF output directory: {output_cohere_dir}")
    print(f"🔢 RRF constant k: {args.rrf_k}")
    
    # Create RRF results
    print("\n🔄 Creating RRF results...")
    stats = create_rrf_results(
        top_qpp_data,
        retrieval_pyserini_dir,
        retrieval_cohere_dir,
        output_pyserini_dir,
        output_cohere_dir,
        k=args.rrf_k
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ RRF Results Creation Complete!")
    print("=" * 70)
    print(f"📊 Queries processed: {len(stats['queries_processed'])}")
    print(f"📊 QPP methods processed: {stats['qpp_methods_processed']}")
    print(f"📁 Pyserini RRF files created: {stats['pyserini_rrf_files']}")
    print(f"📁 Cohere RRF files created: {stats['cohere_rrf_files']}")
    print(f"\n💾 Results saved to:")
    print(f"   - {output_pyserini_dir}")
    print(f"   - {output_cohere_dir}")


if __name__ == '__main__':
    main()
