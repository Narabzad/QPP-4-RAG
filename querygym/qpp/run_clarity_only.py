#!/usr/bin/env python3
"""
Run ONLY Clarity QPP metric on all query+run combinations.
This script focuses on computing Clarity scores that are missing from existing results.
"""

import sys
sys.path.append('/future/u/negara/home/set_based_QPP')

from pyserini.index import LuceneIndexReader
import numpy as np
import argparse
import pytrec_eval
import os
import glob
import math
import time
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from pathlib import Path

# ============================================================================
# CLARITY COMPUTATION FUNCTIONS
# ============================================================================

def RM1(qtokens, pid_list, score_list, index_reader, k, mu=1000):
    """Compute RM1 relevance model"""
    V = []
    doc_len = np.zeros(k)

    for idx_p, pid in enumerate(pid_list[:k]):
        try:
            doc_vector = index_reader.get_document_vector(pid)
            if doc_vector:
                V += list(doc_vector.keys())
                doc_len[idx_p] = sum(doc_vector.values())
        except Exception as e:
            print(f"    Warning: Could not get document vector for {pid}: {e}")
            continue

    V = list(set(V))
    if len(V) == 0:
        return np.array([], dtype=[('tokens', object), ('token_scores', np.float32)])
    
    mat = np.zeros([k, len(V)])

    for idx_p, pid in enumerate(pid_list[:k]):
        try:
            doc_vector = index_reader.get_document_vector(pid)
            if doc_vector:
                for token in doc_vector.keys():
                    if token in V:
                        mat[idx_p, V.index(token)] = doc_vector[token]
        except Exception as e:
            continue

    if len(score_list[:k]) == len(doc_len) and sum(score_list[:k]) > 0:
        _p_w_q = np.dot(np.array([score_list[:k] / (doc_len + 1e-10), ]), mat)
        p_w_q = np.asarray(_p_w_q / sum(score_list[:k])).squeeze()
        rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]), order='token_scores')[::-1]
    else:
        rm1 = np.array([], dtype=[('tokens', object), ('token_scores', np.float32)])
    
    return rm1

def CLARITY(rm1, index_reader, term_num=100):
    """Compute Clarity score from RM1 model"""
    if len(rm1) == 0:
        return 0.0
    
    rm1_cut = rm1[:min(term_num, len(rm1))]
    if len(rm1_cut) == 0:
        return 0.0
    
    p_w_q = rm1_cut['token_scores'] / (rm1_cut['token_scores'].sum() + 1e-10)
    
    # Get collection probabilities for each term
    p_t_D = []
    for token in rm1_cut['tokens']:
        try:
            df, cf = index_reader.get_term_counts(token, analyzer=None)
            if cf > 0:
                p_t_D.append(cf / index_reader.stats()['total_terms'])
            else:
                p_t_D.append(1e-10)  # Small value for unseen terms
        except:
            p_t_D.append(1e-10)
    
    p_t_D = np.array(p_t_D)
    
    # Compute KL divergence
    clarity_score = np.sum(p_w_q * np.log2((p_w_q + 1e-10) / (p_t_D + 1e-10)))
    
    return clarity_score

# ============================================================================
# PROCESSING FUNCTION
# ============================================================================

def process_clarity_for_file(args):
    """Process Clarity for a single query+run combination"""
    query_file, run_file, index_path, output_dir = args
    
    # Extract names
    query_basename = os.path.basename(query_file)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    run_basename = os.path.basename(run_file)
    run_name = run_basename.replace('run.', '').replace('.txt', '')
    
    # Determine retrieval method
    if 'retrieval_cohere' in run_file:
        retrieval_method = 'cohere'
    else:
        retrieval_method = 'pyserini'
    
    # Output file name
    output_file = os.path.join(output_dir, f"post_retrieval_{query_version}_{run_name}_{retrieval_method}_qpp_metrics.csv")
    
    task_name = f"{query_version}+{run_name}({retrieval_method})"
    
    try:
        # Always recompute - don't skip based on existing values
        # (The existing values are likely 0.0 due to computation issues)
        
        # Load index
        index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
        
        # Load run file
        with open(run_file, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)
        
        # Load query file
        query = {}
        with open(query_file, 'r') as f_query:
            for line in f_query:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        query[qid] = qtext
        
        # Compute Clarity for each query
        clarity_results = {}
        errors = []
        
        for qid, qtext in query.items():
            if qid not in run:
                clarity_results[qid] = {'clarity-score-k10': 0.0}
                continue
            
            try:
                qtokens = index_reader.analyze(qtext)
                
                if len(qtokens) == 0:
                    clarity_results[qid] = {'clarity-score-k10': 0.0}
                    continue
                
                # Get top documents and scores
                pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
                score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
                
                if len(pid_list) == 0 or len(score_list) == 0:
                    clarity_results[qid] = {'clarity-score-k10': 0.0}
                    continue
                
                # Compute Clarity with k=10 (faster)
                k_val = min(10, len(score_list))
                rm1_k10 = RM1(qtokens, pid_list, score_list, index_reader, k_val, mu=1000)
                
                if len(rm1_k10) == 0:
                    clarity_results[qid] = {'clarity-score-k10': 0.0}
                else:
                    clarity_k10 = CLARITY(rm1_k10, index_reader, term_num=100)
                    clarity_results[qid] = {'clarity-score-k10': clarity_k10}
                
            except Exception as e:
                errors.append(f"Query {qid}: {str(e)[:50]}")
                clarity_results[qid] = {'clarity-score-k10': 0.0}
        
        # Update existing CSV or create new one
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            
            # Update clarity scores
            for qid, metrics in clarity_results.items():
                if qid in df['query_id'].values:
                    df.loc[df['query_id'] == qid, 'clarity-score-k10'] = metrics['clarity-score-k10']
            
            # Save updated file
            df.to_csv(output_file, index=False)
            
            return f"✓ {task_name}"
        else:
            # File doesn't exist - this shouldn't happen, but handle it
            return f"SKIP: {task_name} (no existing file to update)"
            
    except Exception as e:
        return f"✗ {task_name}: {str(e)[:100]}"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run ONLY Clarity QPP metric on QueryGym dataset (k=10 for speed)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--num_processes', type=int, default=20,
                       help='Number of parallel processes (default: 20)')
    parser.add_argument('--queries_dir', default="/future/u/negara/home/set_based_QPP/querygym/queries",
                       help='Directory containing query files')
    parser.add_argument('--retrieval_dirs', nargs='+', 
                       default=["/future/u/negara/home/set_based_QPP/querygym/retrieval",
                                "/future/u/negara/home/set_based_QPP/querygym/retrieval_cohere"],
                       help='Directories containing retrieval run files')
    parser.add_argument('--index_path', default="msmarco-v2.1-doc-segmented",
                       help='Pyserini prebuilt index name')
    parser.add_argument('--output_dir', default="/future/u/negara/home/set_based_QPP/querygym/qpp",
                       help='Output directory for QPP results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLARITY QPP METRIC COMPUTATION (k=10 for speed)")
    print("="*80)
    print(f"Queries directory: {args.queries_dir}")
    print(f"Retrieval directories: {args.retrieval_dirs}")
    print(f"Index: {args.index_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel processes: {args.num_processes}")
    print("="*80)
    
    # Get all query files
    query_files = glob.glob(os.path.join(args.queries_dir, "topics.*.txt"))
    query_files.sort()
    
    # Prepare arguments for processing
    task_args = []
    
    for query_file in query_files:
        query_basename = os.path.basename(query_file)
        query_version = query_basename.replace('topics.', '').replace('.txt', '')
        
        # Find corresponding run files in each retrieval directory
        for retrieval_dir in args.retrieval_dirs:
            run_file = os.path.join(retrieval_dir, f"run.{query_version}.txt")
            if os.path.exists(run_file):
                task_args.append((query_file, run_file, args.index_path, args.output_dir))
    
    print(f"\nFound {len(query_files)} query files")
    print(f"Found {len(task_args)} query+run combinations to process")
    print("-" * 80)
    
    # Process in parallel
    start_time = time.time()
    results = []
    
    print(f"\n🚀 Starting parallel processing with {args.num_processes} workers...")
    print(f"📊 Processing Clarity (k=10) for all combinations...\n")
    
    with mp.Pool(processes=args.num_processes) as pool:
        with tqdm(total=len(task_args), desc="Computing Clarity", unit="file", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for result in pool.imap_unordered(process_clarity_for_file, task_args):
                results.append(result)
                pbar.update(1)
                
                # Print status for completed tasks
                if result.startswith("✓"):
                    tqdm.write(f"  {result}")
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 CLARITY COMPUTATION COMPLETE!")
    print("="*80)
    
    # Count results
    success = sum(1 for r in results if r.startswith("✓"))
    skipped = sum(1 for r in results if r.startswith("SKIP"))
    failed = sum(1 for r in results if r.startswith("✗"))
    
    print(f"📊 RESULTS:")
    print(f"   ✓ Success: {success}/{len(task_args)}")
    print(f"   ⊘ Skipped: {skipped}/{len(task_args)} (already computed)")
    print(f"   ✗ Failed:  {failed}/{len(task_args)}")
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"📁 Output directory: {args.output_dir}")
    print("="*80)
    
    # Show failed tasks if any
    if failed > 0:
        print("\n⚠️  Failed tasks:")
        for result in results:
            if result.startswith("✗"):
                print(f"   {result}")

if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()
