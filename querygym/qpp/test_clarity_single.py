#!/usr/bin/env python3
"""
Test Clarity computation on a single file to debug issues
"""

import sys
sys.path.append('/future/u/negara/home/set_based_QPP')

from pyserini.index import LuceneIndexReader
import numpy as np
import pytrec_eval
import math

# Test with one file
query_file = "/future/u/negara/home/set_based_QPP/querygym/queries/topics.original.txt"
run_file = "/future/u/negara/home/set_based_QPP/querygym/retrieval/run.original.txt"
index_path = "msmarco-v2.1-doc-segmented"

print("Loading index...")
index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
print(f"✓ Index loaded. Total docs: {index_reader.stats()['documents']}")

print("\nLoading run file...")
with open(run_file, 'r') as f_run:
    run = pytrec_eval.parse_run(f_run)
print(f"✓ Run file loaded. Queries: {len(run)}")

print("\nLoading query file...")
query = {}
with open(query_file, 'r') as f_query:
    for line in f_query:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, qtext = parts[0], parts[1]
                query[qid] = qtext
print(f"✓ Query file loaded. Queries: {len(query)}")

# Test with first query
qid = list(query.keys())[0]
qtext = query[qid]

print(f"\n{'='*80}")
print(f"Testing with query: {qid}")
print(f"Query text: {qtext}")
print(f"{'='*80}")

# Analyze query
qtokens = index_reader.analyze(qtext)
print(f"\nQuery tokens ({len(qtokens)}): {qtokens}")

# Get retrieval results
if qid in run:
    pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
    score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
    
    print(f"\nRetrieved documents: {len(pid_list)}")
    print(f"Top 10 PIDs: {pid_list[:10]}")
    print(f"Top 10 scores: {score_list[:10]}")
    
    # Compute RM1
    print(f"\nComputing RM1 with k=10...")
    k = min(10, len(score_list))
    
    V = []
    doc_len = np.zeros(k)
    
    for idx_p, pid in enumerate(pid_list[:k]):
        try:
            doc_vector = index_reader.get_document_vector(pid)
            if doc_vector:
                V += list(doc_vector.keys())
                doc_len[idx_p] = sum(doc_vector.values())
                print(f"  Doc {idx_p} ({pid}): {len(doc_vector)} terms, length={doc_len[idx_p]}")
            else:
                print(f"  Doc {idx_p} ({pid}): No document vector!")
        except Exception as e:
            print(f"  Doc {idx_p} ({pid}): Error - {e}")
    
    V = list(set(V))
    print(f"\nVocabulary size: {len(V)}")
    
    if len(V) > 0:
        mat = np.zeros([k, len(V)])
        
        for idx_p, pid in enumerate(pid_list[:k]):
            try:
                doc_vector = index_reader.get_document_vector(pid)
                if doc_vector:
                    for token in doc_vector.keys():
                        if token in V:
                            mat[idx_p, V.index(token)] = doc_vector[token]
            except:
                pass
        
        print(f"Matrix shape: {mat.shape}")
        print(f"Doc lengths: {doc_len}")
        print(f"Scores: {score_list[:k]}")
        
        if sum(score_list[:k]) > 0 and sum(doc_len) > 0:
            _p_w_q = np.dot(np.array([score_list[:k] / (doc_len + 1e-10), ]), mat)
            p_w_q = np.asarray(_p_w_q / sum(score_list[:k])).squeeze()
            rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]), order='token_scores')[::-1]
            
            print(f"\nRM1 computed. Terms: {len(rm1)}")
            print(f"Top 10 RM1 terms:")
            for i in range(min(10, len(rm1))):
                print(f"  {rm1[i]['tokens']}: {rm1[i]['token_scores']}")
            
            # Compute Clarity
            print(f"\nComputing Clarity...")
            term_num = 100
            rm1_cut = rm1[:min(term_num, len(rm1))]
            p_w_q_norm = rm1_cut['token_scores'] / (rm1_cut['token_scores'].sum() + 1e-10)
            
            print(f"Using top {len(rm1_cut)} terms")
            print(f"Sum of normalized probabilities: {p_w_q_norm.sum()}")
            
            # Get collection probabilities
            p_t_D = []
            for token in rm1_cut['tokens']:
                try:
                    df, cf = index_reader.get_term_counts(token, analyzer=None)
                    if cf > 0:
                        p_t_D.append(cf / index_reader.stats()['total_terms'])
                    else:
                        p_t_D.append(1e-10)
                except:
                    p_t_D.append(1e-10)
            
            p_t_D = np.array(p_t_D)
            
            print(f"Collection probabilities computed")
            print(f"Sample p_t_D (first 5): {p_t_D[:5]}")
            
            # Compute KL divergence
            clarity_score = np.sum(p_w_q_norm * np.log2((p_w_q_norm + 1e-10) / (p_t_D + 1e-10)))
            
            print(f"\n{'='*80}")
            print(f"CLARITY SCORE: {clarity_score}")
            print(f"{'='*80}")
        else:
            print("\nERROR: Sum of scores or doc lengths is 0!")
    else:
        print("\nERROR: No vocabulary extracted from documents!")
else:
    print(f"\nERROR: Query {qid} not in run file!")
