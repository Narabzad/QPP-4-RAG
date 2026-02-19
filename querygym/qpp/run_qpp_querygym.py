#!/usr/bin/env python3
"""
Modular QPP (Query Performance Prediction) script for QueryGym dataset.

This script runs pre-retrieval and post-retrieval QPP metrics on:
- Query files from: /future/u/negara/home/set_based_QPP/querygym/queries
- Retrieval results from: 
  - /future/u/negara/home/set_based_QPP/querygym/retrieval
  - /future/u/negara/home/set_based_QPP/querygym/retrieval_cohere

Results are saved to: /future/u/negara/home/set_based_QPP/querygym/qpp

Usage:
    # Run all methods
    python run_qpp_querygym.py --mode all
    
    # Run only pre-retrieval methods
    python run_qpp_querygym.py --mode pre
    
    # Run only post-retrieval methods
    python run_qpp_querygym.py --mode post
    
    # Run specific query file(s)
    python run_qpp_querygym.py --mode pre --query_files topics.original.txt
    
    # Run with custom parallelism
    python run_qpp_querygym.py --mode all --num_processes 10
"""

import sys
sys.path.append('/future/u/negara/home/set_based_QPP')

from pyserini.index import LuceneIndexReader
import json
from collections import Counter
import numpy as np
import argparse
import pytrec_eval
import os
import glob
import math
import time
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import threading
from pathlib import Path

# ============================================================================
# QPP METRIC FUNCTIONS (from run_qpp_batch.py)
# ============================================================================

# Pre-retrieval QPP metrics
def IDF(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)
    if df == 0:
        return 0.0
    else:
        return math.log2(index_reader.stats()['documents']/df)

def SCQ(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)
    if cf == 0:
        return 0.0
    else:
        part_A = 1 + math.log2(cf)
        part_B = IDF(term, index_reader)
        return part_A * part_B

def avg_max_sum_std_IDF(qtokens, index_reader):
    v = []
    for t in qtokens:
        v.append(IDF(t, index_reader))
    return [np.mean(v), max(v), sum(v), np.std(v)]

def avg_max_sum_SCQ(qtokens, index_reader):
    scq = []
    for t in qtokens:
        scq.append(SCQ(t, index_reader))
    return [np.mean(scq), max(scq), sum(scq)]

def ictf(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)
    if cf == 0:
        return 0.0
    else:
        return math.log2(index_reader.stats()['total_terms']/cf)

def avgICTF(qtokens, index_reader):
    v = []
    for t in qtokens:
        v.append(ictf(t, index_reader))
    return np.mean(v)

def SCS_1(qtokens, index_reader):
    part_A = math.log2(1/len(qtokens))
    part_B = avgICTF(qtokens, index_reader)
    return part_A + part_B

def SCS_2(qtokens, index_reader):
    v = []
    qtf = Counter(qtokens)
    ql = len(qtokens)
    
    for t in qtokens:
        pml = qtf[t]/ql
        df, cf = index_reader.get_term_counts(t, analyzer=None)
        pcoll = cf / index_reader.stats()['total_terms']
        
        if pcoll == 0:
            v.append(0.0)
        else:
            v.append(pml*math.log2(pml/pcoll))
    
    return sum(v)

def t2did(t, index_reader):
    postings_list = index_reader.get_postings_list(t, analyzer=None)
    if postings_list == None:
        return []
    else:
        return [posting.docid for posting in postings_list]

def QS(qtokens, qtoken2did, index_reader):
    q2did_set = set()
    for t in qtokens:
        q2did_set = q2did_set.union(set(qtoken2did.get(t, [])))
    
    n_Q = len(q2did_set)
    N = index_reader.stats()['documents']
    return -math.log2(n_Q/N)

def VAR(t, index_reader):
    postings_list = index_reader.get_postings_list(t, analyzer=None)
    if postings_list == None:
        return 0.0, 0.0
    else:
        tf_array = np.array([posting.tf for posting in postings_list])
        tf_idf_array = np.log2(1 + tf_array) * IDF(t, index_reader)
        return np.var(tf_idf_array), np.std(tf_idf_array)

def avg_max_sum_VAR(qtokens, qtoken2var):
    v = []
    for t in qtokens:
        v.append(qtoken2var.get(t, 0.0))
    return [np.mean(v), max(v), sum(v)]

def PMI(t_i, t_j, index_reader, qtoken2did):
    titj_doc_count = len(set(qtoken2did.get(t_i, [])).intersection(set(qtoken2did.get(t_j, []))))
    ti_doc_count = len(qtoken2did.get(t_i, []))
    tj_doc_count = len(qtoken2did.get(t_j, []))
    
    if titj_doc_count > 0:
        part_A = titj_doc_count/index_reader.stats()['documents']
        part_B = (ti_doc_count/index_reader.stats()['documents'])*(tj_doc_count/index_reader.stats()['documents'])
        return math.log2(part_A/part_B)
    else:
        return 0.0

def avg_max_sum_PMI(qtokens, index_reader, qtoken2did):
    pair = []
    pair_num = 0
    
    if len(qtokens) == 0:
        return [0.0, 0.0, 0.0]
    else:
        for i in range(0, len(qtokens)):
            for j in range(i + 1, len(qtokens)):
                pair_num += 1
                pair.append(PMI(qtokens[i], qtokens[j], index_reader, qtoken2did))
        
        assert len(pair) == pair_num
        return [np.mean(pair), max(pair), sum(pair)]

# Post-retrieval QPP metrics
def RM1(qtokens, pid_list, score_list, index_reader, k, mu=1000):
    V = []
    doc_len = np.zeros(k)

    for idx_p, pid in enumerate(pid_list[:k]):
        V += index_reader.get_document_vector(pid).keys()
        doc_len[idx_p] = sum(index_reader.get_document_vector(pid).values())

    V = list(set(V))
    mat = np.zeros([k, len(V)])

    for idx_p, pid in enumerate(pid_list[:k]):
        for token in index_reader.get_document_vector(pid).keys():
            mat[idx_p, V.index(token)] = index_reader.get_document_vector(pid)[token]

    if len(score_list[:k]) == len(doc_len):
        _p_w_q = np.dot(np.array([score_list[:k] / doc_len , ]), mat)
        p_w_q = np.asarray(_p_w_q/ sum(score_list[:k])).squeeze()
        rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]), order='token_scores')[::-1]
    else:
        rm1 = np.array([], dtype=[('tokens', np.object), ('token_scores', np.float32)])
    return rm1

def CLARITY(rm1, index_reader, term_num=100):
    if len(rm1) == 0:
        return 0.0
    
    rm1_cut = rm1[:term_num]
    p_w_q = rm1_cut['token_scores'] / rm1_cut['token_scores'].sum()
    p_t_D = np.array([[index_reader.get_term_counts(token, analyzer=None)[1] for token in rm1_cut['tokens']], ]) / index_reader.stats()['total_terms']
    return np.log(p_w_q / p_t_D).dot(p_w_q)[0]

def WIG(qtokens, score_list, k):
    if len(score_list) == 0:
        return 0.0, 0.0
    
    corpus_score = np.mean(score_list)
    wig_norm = (np.mean(score_list[:k]) - corpus_score)/ np.sqrt(len(qtokens))
    wig_no_norm = np.mean(score_list[:k]) / np.sqrt(len(qtokens))
    return wig_norm, wig_no_norm

def NQC(score_list, k):
    if len(score_list) == 0:
        return 0.0, 0.0
    
    corpus_score = np.mean(score_list)
    nqc_norm = np.std(score_list[:k]) / corpus_score
    nqc_no_norm = np.std(score_list[:k])
    return nqc_norm, nqc_no_norm

def SIGMA_MAX(score_list):
    if len(score_list) == 0:
        return 0.0, 0
    
    max_std = 0
    scores = []
    
    for idx, score in enumerate(score_list):
        scores.append(score)
        if np.std(scores) > max_std:
            max_std = np.std(scores)
    
    return max_std, len(scores)

def SIGMA_X(qtokens, score_list, x):
    if len(score_list) == 0:
        return 0.0, 0
    
    top_score = score_list[0]
    scores = []
    
    for idx, score in enumerate(score_list):
        if score >= (top_score*x):
            scores.append(score)
    
    return np.std(scores)/np.sqrt(len(qtokens)), len(scores)

def SMV(score_list, k):
    if len(score_list) == 0:
        return 0.0, 0.0
    
    corpus_score = np.mean(score_list)
    mu = np.mean(score_list[:k])
    smv_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))/corpus_score
    smv_no_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))
    return smv_norm, smv_no_norm

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_pre_retrieval_metrics(query_file_path, index_path, output_dir):
    """Process pre-retrieval QPP metrics for a single query file - EXACT COPY from run_qpp_batch.py"""
    
    # Extract query version for output file
    query_basename = os.path.basename(query_file_path)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    
    print(f"Processing pre-retrieval metrics for {query_version}...")
    
    # Load index
    try:
        index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
        print(f"✓ Loaded index: {index_path}")
    except Exception as e:
        print(f"✗ Failed to load index: {e}")
        return None
    
    # Load query file
    try:
        query = {}
        with open(query_file_path, 'r') as f_query:
            for line in f_query:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        query[qid] = qtext
        print(f"✓ Loaded {len(query)} queries from {query_file_path}")
    except Exception as e:
        print(f"✗ Failed to load query file: {e}")
        return None
    
    # Pre-compute token mappings for efficiency
    print("  Pre-computing token mappings...")
    qtoken_set = set()
    
    # First pass: collect all unique tokens
    print("    Collecting unique tokens...")
    for qtext in tqdm(query.values(), desc="Analyzing queries"):
        qtokens = index_reader.analyze(qtext)
        for qtoken in qtokens:
            qtoken_set.add(qtoken)
    
    print(f"    Found {len(qtoken_set)} unique tokens")
    
    qtoken2var = {}
    qtoken2std = {}
    qtoken2did = {}
    
    # Second pass: compute statistics for each token
    print("    Computing token statistics...")
    for qtoken in tqdm(qtoken_set, desc="Processing tokens"):
        qtoken2var[qtoken], qtoken2std[qtoken] = VAR(qtoken, index_reader)
        qtoken2did[qtoken] = t2did(qtoken, index_reader)
    
    # Compute pre-retrieval QPP metrics for each query
    predicted_performance = {}
    
    print("    Computing QPP metrics for queries...")
    for qid, qtext in tqdm(query.items(), desc="Processing queries"):
        predicted_performance[qid] = {}
        
        try:
            qtokens = index_reader.analyze(qtext)
            
            # === PRE-RETRIEVAL METRICS ===
            
            # QS (Query Scope)
            predicted_performance[qid]["QS"] = QS(qtokens, qtoken2did, index_reader)
            
            # PMI (Pointwise Mutual Information)
            predicted_performance[qid]["PMI-avg"], predicted_performance[qid]["PMI-max"], predicted_performance[qid]["PMI-sum"] = avg_max_sum_PMI(qtokens, index_reader, qtoken2did)
            
            # Query length
            predicted_performance[qid]["ql"] = len(qtokens)
            
            # VAR (Variance)
            predicted_performance[qid]["VAR-var-avg"], predicted_performance[qid]["VAR-var-max"], predicted_performance[qid]["VAR-var-sum"] = avg_max_sum_VAR(qtokens, qtoken2var)
            predicted_performance[qid]["VAR-std-avg"], predicted_performance[qid]["VAR-std-max"], predicted_performance[qid]["VAR-std-sum"] = avg_max_sum_VAR(qtokens, qtoken2std)
            
            # IDF (Inverse Document Frequency)
            predicted_performance[qid]["IDF-avg"], predicted_performance[qid]["IDF-max"], predicted_performance[qid]["IDF-sum"], predicted_performance[qid]["IDF-std"] = avg_max_sum_std_IDF(qtokens, index_reader)
            
            # SCQ (Sum Collection Query)
            predicted_performance[qid]["SCQ-avg"], predicted_performance[qid]["SCQ-max"], predicted_performance[qid]["SCQ-sum"] = avg_max_sum_SCQ(qtokens, index_reader)
            
            # avgICTF (Average Inverse Collection Term Frequency)
            predicted_performance[qid]["avgICTF"] = avgICTF(qtokens, index_reader)
            
            # SCS (Simplified Collection Statistics)
            predicted_performance[qid]["SCS-1"] = SCS_1(qtokens, index_reader)
            predicted_performance[qid]["SCS-2"] = SCS_2(qtokens, index_reader)
            
        except Exception as e:
            print(f"  ✗ Error processing query {qid}: {e}")
            continue
    
    if not predicted_performance:
        print(f"  ✗ No valid predictions generated for {query_version}")
        return None
    
    # Get metric names
    metric_names = list(next(iter(predicted_performance.values())).keys())
    
    # Create output file
    output_file = os.path.join(output_dir, f"pre_retrieval_{query_version}_qpp_metrics.csv")
    
    try:
        with open(output_file, 'w') as f:
            # Write header
            header = "query_id," + ",".join(metric_names) + "\n"
            f.write(header)
            
            # Write data
            for qid, metrics in predicted_performance.items():
                row = f"{qid}"
                for metric in metric_names:
                    row += f",{metrics.get(metric, 'N/A')}"
                row += "\n"
                f.write(row)
        
        print(f"  ✓ Saved pre-retrieval QPP metrics to {output_file}")
        print(f"SUCCESS: {output_file}")  # For subprocess parsing
        return output_file
        
    except Exception as e:
        print(f"  ✗ Failed to save output: {e}")
        return None

def process_post_retrieval_metrics(query_file_path, run_file_path, index_path, output_dir, k_top=100):
    """Process post-retrieval QPP metrics for a query file and run file combination"""
    
    # Extract names for output file
    query_basename = os.path.basename(query_file_path)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    run_basename = os.path.basename(run_file_path)
    run_name = run_basename.replace('run.', '').replace('.txt', '')
    
    # Determine retrieval method from directory
    if 'retrieval_cohere' in run_file_path:
        retrieval_method = 'cohere'
    else:
        retrieval_method = 'pyserini'
    
    print(f"Processing post-retrieval metrics for {query_version} with {run_name} ({retrieval_method})...")
    
    # Load index
    try:
        index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
        print(f"✓ Loaded index: {index_path}")
    except Exception as e:
        print(f"✗ Failed to load index: {e}")
        return None
    
    # Load run file
    try:
        with open(run_file_path, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)
        print(f"✓ Loaded run file: {run_file_path}")
    except Exception as e:
        print(f"✗ Failed to load run file: {e}")
        return None
    
    # Load query file
    try:
        query = {}
        with open(query_file_path, 'r') as f_query:
            for line in f_query:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        query[qid] = qtext
        print(f"✓ Loaded {len(query)} queries from {query_file_path}")
    except Exception as e:
        print(f"✗ Failed to load query file: {e}")
        return None
    
    # Compute post-retrieval QPP metrics for each query
    predicted_performance = {}
    
    print("    Computing QPP metrics for queries...")
    for qid, qtext in tqdm(query.items(), desc="Processing queries"):
        if qid not in run:
            continue
            
        predicted_performance[qid] = {}
        
        try:
            qtokens = index_reader.analyze(qtext)
            
            # Get top k documents and scores
            pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
            score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
            
            # Limit to top k
            pid_list = pid_list[:k_top]
            score_list = score_list[:k_top]
            
            if len(score_list) == 0:
                continue
            
            # === POST-RETRIEVAL METRICS ===
            
            # CLARITY with different k values
            rm1_k10 = RM1(qtokens, pid_list, score_list, index_reader, min(10, len(score_list)), mu=1000)
            predicted_performance[qid]["clarity-score-k10"] = CLARITY(rm1_k10, index_reader, term_num=100)
            del rm1_k10
            
            rm1_k100 = RM1(qtokens, pid_list, score_list, index_reader, min(100, len(score_list)), mu=1000)
            predicted_performance[qid]["clarity-score-k100"] = CLARITY(rm1_k100, index_reader, term_num=100)
            del rm1_k100
            
            # WIG (Weighted Information Gain)
            predicted_performance[qid]["wig-norm-k100"], predicted_performance[qid]["wig-no-norm-k100"] = WIG(qtokens, score_list, min(100, len(score_list)))
            predicted_performance[qid]["wig-norm-k1000"], predicted_performance[qid]["wig-no-norm-k1000"] = WIG(qtokens, score_list, min(1000, len(score_list)))
            
            # NQC (Normalized Query Commitment)
            predicted_performance[qid]["nqc-norm-k100"], predicted_performance[qid]["nqc-no-norm-k100"] = NQC(score_list, min(100, len(score_list)))
            
            # SMV (Score Mean Variance)
            predicted_performance[qid]["smv-norm-k100"], predicted_performance[qid]["smv-no-norm-k100"] = SMV(score_list, min(100, len(score_list)))
            
            # SIGMA metrics
            predicted_performance[qid]["sigma-x0.5"], _ = SIGMA_X(qtokens, score_list, 0.5)
            predicted_performance[qid]["sigma-max"], _ = SIGMA_MAX(score_list)
            
            # RSD (Relative Score Deviation)
            _, predicted_performance[qid]["RSD"] = SMV(score_list, min(1000, len(score_list)))
            
        except Exception as e:
            print(f"  ✗ Error processing query {qid}: {e}")
            continue
    
    if not predicted_performance:
        print(f"  ✗ No valid predictions generated for {query_version} with {run_name}")
        return None
    
    # Get metric names
    metric_names = list(next(iter(predicted_performance.values())).keys())
    
    # Create output file with retrieval method in name
    output_file = os.path.join(output_dir, f"post_retrieval_{query_version}_{run_name}_{retrieval_method}_qpp_metrics.csv")
    
    try:
        # Write to temporary file first, then rename (atomic write)
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            # Write header
            header = "query_id," + ",".join(metric_names) + "\n"
            f.write(header)
            f.flush()  # Ensure header is written
            
            # Write data
            for qid, metrics in predicted_performance.items():
                row = f"{qid}"
                for metric in metric_names:
                    row += f",{metrics.get(metric, 'N/A')}"
                row += "\n"
                f.write(row)
                f.flush()  # Flush after each row to ensure immediate save
        
        # Atomic rename
        os.rename(temp_file, output_file)
        
        # Flush and sync to ensure file is written to disk
        os.sync()
        
        print(f"  ✓ Saved post-retrieval QPP metrics to {output_file}")
        print(f"  📁 FILE CREATED: {os.path.basename(output_file)}", flush=True)
        return output_file
        
    except Exception as e:
        print(f"  ✗ Failed to save output: {e}")
        return None

# ============================================================================
# PARALLEL PROCESSING WRAPPERS (using subprocess to avoid JVM conflicts)
# ============================================================================

def run_pre_retrieval_subprocess(args):
    """Run pre-retrieval processing in subprocess to avoid JVM conflicts"""
    query_file, index_path, output_dir = args
    query_basename = os.path.basename(query_file)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    
    script_path = os.path.join(os.path.dirname(__file__), 'process_single_qpp.py')
    
    cmd = [
        sys.executable, script_path,
        '--mode', 'pre',
        '--query_file', query_file,
        '--index_path', index_path,
        '--output_dir', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout (pre-retrieval is slow)
        
        if result.returncode == 0:
            # Extract filename from stdout
            for line in result.stdout.strip().split('\n'):
                if line.startswith('SUCCESS:'):
                    output_file = line.replace('SUCCESS: ', '').strip()
                    # Verify file exists
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"✅ COMPLETED: Pre-retrieval {query_version} → {os.path.basename(output_file)} ({file_size} bytes)", flush=True)
                        return output_file
                    else:
                        print(f"⚠️  WARNING: File not found after creation: {output_file}")
            return None
        else:
            print(f"❌ FAILED: Pre-retrieval {query_version}")
            if result.stderr:
                print(f"   Stderr: {result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: Pre-retrieval {query_version} (30 minutes)")
        return None
    except Exception as e:
        print(f"💥 ERROR: Pre-retrieval {query_version} - {e}")
        return None

def run_post_retrieval_subprocess(args):
    """Run post-retrieval processing in subprocess to avoid JVM conflicts"""
    query_file, run_file, index_path, output_dir, k_top = args
    query_basename = os.path.basename(query_file)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    run_basename = os.path.basename(run_file)
    run_name = run_basename.replace('run.', '').replace('.txt', '')
    
    script_path = os.path.join(os.path.dirname(__file__), 'process_single_qpp.py')
    
    cmd = [
        sys.executable, script_path,
        '--mode', 'post',
        '--query_file', query_file,
        '--run_file', run_file,
        '--index_path', index_path,
        '--output_dir', output_dir,
        '--k_top', str(k_top)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Extract filename from stdout
            for line in result.stdout.strip().split('\n'):
                if line.startswith('SUCCESS:'):
                    output_file = line.replace('SUCCESS: ', '').strip()
                    # Verify file exists
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"✅ COMPLETED: {query_version} + {run_name} → {os.path.basename(output_file)} ({file_size} bytes)", flush=True)
                        return output_file
                    else:
                        print(f"⚠️  WARNING: File not found after creation: {output_file}")
            return None
        else:
            print(f"❌ FAILED: {query_version} + {run_name}")
            if result.stderr:
                print(f"   Stderr: {result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {query_version} + {run_name} (1 hour)")
        return None
    except Exception as e:
        print(f"💥 ERROR: {query_version} + {run_name} - {e}")
        return None

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def run_pre_retrieval(queries_dir, index_path, output_dir, num_processes, query_files_filter=None):
    """Run pre-retrieval QPP metrics on all query files"""
    
    print("\n" + "="*60)
    print("PRE-RETRIEVAL QPP METRICS")
    print("="*60)
    
    # Get all query files
    query_files = glob.glob(os.path.join(queries_dir, "topics.*.txt"))
    query_files.sort()
    
    if query_files_filter:
        # Filter to only specified files
        query_files = [f for f in query_files if any(filter_name in os.path.basename(f) for filter_name in query_files_filter)]
    
    if not query_files:
        print(f"No query files found in {queries_dir}")
        return []
    
    print(f"Found {len(query_files)} query files to process")
    print(f"Using index: {index_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Prepare arguments
    pre_retrieval_args = [(query_file, index_path, output_dir) for query_file in query_files]
    
    # Process in parallel
    start_time = time.time()
    results = []
    
    if num_processes == 1:
        # Sequential processing
        for args in tqdm(pre_retrieval_args, desc="Pre-retrieval processing"):
            result = run_pre_retrieval_subprocess(args)
            if result:
                results.append(result)
    else:
        # Parallel processing using subprocess (avoids JVM conflicts)
        print(f"🚀 Launching {num_processes} parallel subprocess workers...")
        print(f"📊 Progress will be saved incrementally - check output directory for files as they complete")
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=len(pre_retrieval_args), desc="Pre-retrieval processing", unit="task") as pbar:
                for result in pool.imap_unordered(run_pre_retrieval_subprocess, pre_retrieval_args):
                    if result:
                        results.append(result)
                        # Print progress update
                        completed = len(results)
                        total = len(pre_retrieval_args)
                        print(f"\n📈 Progress: {completed}/{total} pre-retrieval files completed", flush=True)
                    pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Pre-retrieval completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Success: {len(results)}/{len(query_files)} files")
    
    return results

def run_post_retrieval(queries_dir, retrieval_dirs, index_path, output_dir, num_processes, k_top, query_files_filter=None):
    """Run post-retrieval QPP metrics on query files and retrieval results"""
    
    print("\n" + "="*60)
    print("POST-RETRIEVAL QPP METRICS")
    print("="*60)
    
    # Get all query files
    query_files = glob.glob(os.path.join(queries_dir, "topics.*.txt"))
    query_files.sort()
    
    if query_files_filter:
        # Filter to only specified files
        query_files = [f for f in query_files if any(filter_name in os.path.basename(f) for filter_name in query_files_filter)]
    
    if not query_files:
        print(f"No query files found in {queries_dir}")
        return []
    
    # Prepare arguments for post-retrieval processing
    post_retrieval_args = []
    
    for query_file in query_files:
        query_basename = os.path.basename(query_file)
        query_version = query_basename.replace('topics.', '').replace('.txt', '')
        
        # Find corresponding run files in each retrieval directory
        for retrieval_dir in retrieval_dirs:
            run_file = os.path.join(retrieval_dir, f"run.{query_version}.txt")
            if os.path.exists(run_file):
                post_retrieval_args.append((query_file, run_file, index_path, output_dir, k_top))
            else:
                print(f"⚠️  Run file not found: {run_file}")
    
    if not post_retrieval_args:
        print(f"No matching run files found in {retrieval_dirs}")
        return []
    
    print(f"Found {len(query_files)} query files")
    print(f"Found {len(post_retrieval_args)} query+run combinations to process")
    print(f"Using index: {index_path}")
    print(f"Processing top {k_top} retrieved items")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Process in parallel
    start_time = time.time()
    results = []
    
    if num_processes == 1:
        # Sequential processing
        for args in tqdm(post_retrieval_args, desc="Post-retrieval processing"):
            result = run_post_retrieval_subprocess(args)
            if result:
                results.append(result)
    else:
        # Parallel processing using subprocess (avoids JVM conflicts)
        print(f"🚀 Launching {num_processes} parallel subprocess workers...")
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=len(post_retrieval_args), desc="Post-retrieval processing", unit="task") as pbar:
                for result in pool.imap_unordered(run_post_retrieval_subprocess, post_retrieval_args):
                    if result:
                        results.append(result)
                    pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Post-retrieval completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Success: {len(results)}/{len(post_retrieval_args)} combinations")
    
    return results

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run QPP metrics on QueryGym dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods
  python run_qpp_querygym.py --mode all
  
  # Run only pre-retrieval
  python run_qpp_querygym.py --mode pre
  
  # Run only post-retrieval
  python run_qpp_querygym.py --mode post
  
  # Run specific query files
  python run_qpp_querygym.py --mode pre --query_files topics.original.txt topics.genqr_trial1.txt
  
  # Custom parallelism
  python run_qpp_querygym.py --mode all --num_processes 10
        """
    )
    
    parser.add_argument('--mode', choices=['all', 'pre', 'post'], default='all',
                       help='Which QPP methods to run: all, pre (pre-retrieval only), or post (post-retrieval only)')
    parser.add_argument('--num_processes', type=int, default=10,
                       help=f'Number of parallel processes per task type (default: 10) - uses subprocess to avoid JVM conflicts')
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
    parser.add_argument('--k_top', type=int, default=100,
                       help='Number of top retrieved items to process (default: 100)')
    parser.add_argument('--query_files', nargs='+', default=None,
                       help='Specific query files to process (e.g., topics.original.txt)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass  # Method already set
    
    print("="*60)
    print("QPP METRICS FOR QUERYGYM DATASET")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Queries directory: {args.queries_dir}")
    print(f"Retrieval directories: {args.retrieval_dirs}")
    print(f"Index: {args.index_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel processes: {args.num_processes}")
    if args.query_files:
        print(f"Query files filter: {args.query_files}")
    print("="*60)
    
    start_time = time.time()
    pre_results = []
    post_results = []
    
    # Run pre and post retrieval in parallel if mode is 'all'
    if args.mode == 'all':
        pre_thread = None
        post_thread = None
        
        # Start pre-retrieval in a thread
        def run_pre():
            nonlocal pre_results
            pre_results = run_pre_retrieval(
                args.queries_dir,
                args.index_path,
                args.output_dir,
                args.num_processes,
                args.query_files
            )
        pre_thread = threading.Thread(target=run_pre, daemon=False)
        pre_thread.start()
        
        # Start post-retrieval in a thread (runs in parallel with pre)
        def run_post():
            nonlocal post_results
            post_results = run_post_retrieval(
                args.queries_dir,
                args.retrieval_dirs,
                args.index_path,
                args.output_dir,
                args.num_processes,
                args.k_top,
                args.query_files
            )
        post_thread = threading.Thread(target=run_post, daemon=False)
        post_thread.start()
        
        # Wait for both to complete
        print("\n⏳ Waiting for pre and post-retrieval to complete in parallel...")
        if pre_thread:
            pre_thread.join()
            print("✓ Pre-retrieval thread completed")
        if post_thread:
            post_thread.join()
            print("✓ Post-retrieval thread completed")
    
    else:
        # Run sequentially for 'pre' or 'post' mode
        # Run pre-retrieval metrics
        if args.mode == 'pre':
            pre_results = run_pre_retrieval(
                args.queries_dir,
                args.index_path,
                args.output_dir,
                args.num_processes,
                args.query_files
            )
        
        # Run post-retrieval metrics
        if args.mode == 'post':
            post_results = run_post_retrieval(
                args.queries_dir,
                args.retrieval_dirs,
                args.index_path,
                args.output_dir,
                args.num_processes,
                args.k_top,
                args.query_files
            )
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 RESULTS SUMMARY:")
    if args.mode in ['all', 'pre']:
        print(f"   • Pre-retrieval:  {len(pre_results)} files generated")
    if args.mode in ['all', 'post']:
        print(f"   • Post-retrieval: {len(post_results)} files generated")
    print(f"⏱️  TIMING:")
    print(f"   • Total time: {total_time/60:.1f} minutes")
    print(f"📁 OUTPUT DIRECTORY: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
