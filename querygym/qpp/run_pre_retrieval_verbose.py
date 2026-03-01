#!/usr/bin/env python3
"""
Verbose pre-retrieval QPP script that logs every query being processed.
This helps verify that processing is happening and not stuck.
"""

import sys
import argparse
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from pyserini.index import LuceneIndexReader
from collections import Counter
import numpy as np
import os
import glob
import math
import time
from datetime import datetime

# Copy all QPP functions from run_qpp_batch.py
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

def process_single_query_file(query_file_path, index_path, output_dir):
    """Process one query file with verbose logging"""
    
    query_basename = os.path.basename(query_file_path)
    query_version = query_basename.replace('topics.', '').replace('.txt', '')
    
    log_prefix = f"[{query_version}]"
    print(f"\n{'='*60}")
    print(f"{log_prefix} STARTING: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Load index
    print(f"{log_prefix} Loading index...")
    start_time = time.time()
    try:
        index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
        print(f"{log_prefix} ✓ Index loaded in {time.time()-start_time:.1f}s")
    except Exception as e:
        print(f"{log_prefix} ✗ Failed to load index: {e}")
        return None
    
    # Load queries
    print(f"{log_prefix} Loading queries...")
    try:
        query = {}
        with open(query_file_path, 'r') as f_query:
            for line in f_query:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        query[qid] = qtext
        print(f"{log_prefix} ✓ Loaded {len(query)} queries")
    except Exception as e:
        print(f"{log_prefix} ✗ Failed to load queries: {e}")
        return None
    
    # Collect unique tokens
    print(f"{log_prefix} Collecting unique tokens...")
    start_time = time.time()
    qtoken_set = set()
    for qtext in query.values():
        qtokens = index_reader.analyze(qtext)
        qtoken_set.update(qtokens)
    print(f"{log_prefix} ✓ Found {len(qtoken_set)} unique tokens in {time.time()-start_time:.1f}s")
    
    # Compute token statistics
    print(f"{log_prefix} Computing token statistics for {len(qtoken_set)} tokens...")
    start_time = time.time()
    qtoken2var = {}
    qtoken2std = {}
    qtoken2did = {}
    
    token_count = 0
    for qtoken in qtoken_set:
        token_count += 1
        if token_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = token_count / elapsed
            remaining = (len(qtoken_set) - token_count) / rate / 60
            print(f"{log_prefix}   Token {token_count}/{len(qtoken_set)} ({token_count/len(qtoken_set)*100:.1f}%) - {rate:.1f} tokens/s - ETA: {remaining:.1f} min", flush=True)
        
        qtoken2var[qtoken], qtoken2std[qtoken] = VAR(qtoken, index_reader)
        qtoken2did[qtoken] = t2did(qtoken, index_reader)
    
    print(f"{log_prefix} ✓ Token statistics computed in {time.time()-start_time:.1f}s")
    
    # Process each query
    print(f"{log_prefix} Computing QPP metrics for {len(query)} queries...")
    predicted_performance = {}
    query_count = 0
    start_time = time.time()
    
    for qid, qtext in query.items():
        query_count += 1
        print(f"{log_prefix}   Query {query_count}/{len(query)}: {qid}", flush=True)
        
        predicted_performance[qid] = {}
        
        try:
            qtokens = index_reader.analyze(qtext)
            
            # Compute all metrics
            predicted_performance[qid]["QS"] = QS(qtokens, qtoken2did, index_reader)
            predicted_performance[qid]["PMI-avg"], predicted_performance[qid]["PMI-max"], predicted_performance[qid]["PMI-sum"] = avg_max_sum_PMI(qtokens, index_reader, qtoken2did)
            predicted_performance[qid]["ql"] = len(qtokens)
            predicted_performance[qid]["VAR-var-avg"], predicted_performance[qid]["VAR-var-max"], predicted_performance[qid]["VAR-var-sum"] = avg_max_sum_VAR(qtokens, qtoken2var)
            predicted_performance[qid]["VAR-std-avg"], predicted_performance[qid]["VAR-std-max"], predicted_performance[qid]["VAR-std-sum"] = avg_max_sum_VAR(qtokens, qtoken2std)
            predicted_performance[qid]["IDF-avg"], predicted_performance[qid]["IDF-max"], predicted_performance[qid]["IDF-sum"], predicted_performance[qid]["IDF-std"] = avg_max_sum_std_IDF(qtokens, index_reader)
            predicted_performance[qid]["SCQ-avg"], predicted_performance[qid]["SCQ-max"], predicted_performance[qid]["SCQ-sum"] = avg_max_sum_SCQ(qtokens, index_reader)
            predicted_performance[qid]["avgICTF"] = avgICTF(qtokens, index_reader)
            predicted_performance[qid]["SCS-1"] = SCS_1(qtokens, index_reader)
            predicted_performance[qid]["SCS-2"] = SCS_2(qtokens, index_reader)
            
            print(f"{log_prefix}     ✓ Query {qid} completed", flush=True)
            
        except Exception as e:
            print(f"{log_prefix}     ✗ Query {qid} failed: {e}")
            continue
    
    print(f"{log_prefix} ✓ All queries processed in {time.time()-start_time:.1f}s")
    
    # Save results
    metric_names = list(next(iter(predicted_performance.values())).keys())
    output_file = os.path.join(output_dir, f"pre_retrieval_{query_version}_qpp_metrics.csv")
    
    print(f"{log_prefix} Saving to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            header = "query_id," + ",".join(metric_names) + "\n"
            f.write(header)
            
            for qid, metrics in predicted_performance.items():
                row = f"{qid}"
                for metric in metric_names:
                    row += f",{metrics.get(metric, 'N/A')}"
                row += "\n"
                f.write(row)
        
        print(f"{log_prefix} ✓ SAVED: {output_file}")
        print(f"SUCCESS: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"{log_prefix} ✗ Failed to save: {e}")
        return None

def main():
    _here = _Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Verbose pre-retrieval QPP processing')
    parser.add_argument('--queries-dir', type=str,
                       default=str(_here.parent / "queries"),
                       help='Directory containing query files')
    parser.add_argument('--output-dir', type=str,
                       default=str(_here),
                       help='Output directory for QPP results')
    parser.add_argument('--index', type=str, default='msmarco-v2.1-doc-segmented',
                       help='Pyserini prebuilt index name')
    args = parser.parse_args()

    queries_dir = args.queries_dir
    index_path = args.index
    output_dir = args.output_dir

    # Get all query files
    query_files = glob.glob(os.path.join(queries_dir, "topics.*.txt"))
    query_files.sort()
    
    print(f"{'='*60}")
    print(f"VERBOSE PRE-RETRIEVAL QPP PROCESSING")
    print(f"{'='*60}")
    print(f"Query files: {len(query_files)}")
    print(f"Index: {index_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Process ONE file at a time (no parallelism for debugging)
    for i, query_file in enumerate(query_files, 1):
        print(f"\n{'#'*60}")
        print(f"FILE {i}/{len(query_files)}: {os.path.basename(query_file)}")
        print(f"{'#'*60}")
        
        result = process_single_query_file(query_file, index_path, output_dir)
        
        if result:
            print(f"✅ File {i}/{len(query_files)} completed: {os.path.basename(result)}")
        else:
            print(f"❌ File {i}/{len(query_files)} failed")
    
    print(f"\n{'='*60}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
