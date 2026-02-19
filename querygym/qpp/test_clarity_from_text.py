#!/usr/bin/env python3
"""
Test Clarity computation from document text on a single query
"""

import sys
sys.path.append('/future/u/negara/home/set_based_QPP')

from pyserini.index import LuceneIndexReader
import numpy as np
import pytrec_eval
from collections import Counter

query_file = "/future/u/negara/home/set_based_QPP/querygym/queries/topics.original.txt"
run_file = "/future/u/negara/home/set_based_QPP/querygym/retrieval/run.original.txt"
index_path = "msmarco-v2.1-doc-segmented"

print("Loading index...")
index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
print(f"✓ Index loaded")

print("\nLoading run and query files...")
with open(run_file, 'r') as f_run:
    run = pytrec_eval.parse_run(f_run)

query = {}
with open(query_file, 'r') as f_query:
    for line in f_query:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, qtext = parts[0], parts[1]
                query[qid] = qtext

# Test with first query
qid = list(query.keys())[0]
qtext = query[qid]

print(f"\nTesting query: {qid}")
print(f"Query text: {qtext}")

qtokens = index_reader.analyze(qtext)
print(f"Query tokens: {qtokens}")

# Get retrieval results
pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]

print(f"\nProcessing top 10 documents...")
k = 10
V = []
doc_len = np.zeros(k)
doc_term_freqs = []

for idx_p, pid in enumerate(pid_list[:k]):
    try:
        doc = index_reader.doc(pid)
        if doc:
            raw_text = doc.raw()
            terms = index_reader.analyze(raw_text)
            tf_dict = dict(Counter(terms))
            
            doc_term_freqs.append(tf_dict)
            V += list(tf_dict.keys())
            doc_len[idx_p] = sum(tf_dict.values())
            
            print(f"  Doc {idx_p}: {len(tf_dict)} unique terms, {doc_len[idx_p]} total terms")
        else:
            doc_term_freqs.append({})
    except Exception as e:
        print(f"  Doc {idx_p}: Error - {e}")
        doc_term_freqs.append({})

V = list(set(V))
print(f"\nVocabulary size: {len(V)}")

if len(V) > 0:
    mat = np.zeros([k, len(V)])
    
    for idx_p, tf_dict in enumerate(doc_term_freqs):
        for token, freq in tf_dict.items():
            if token in V:
                mat[idx_p, V.index(token)] = freq
    
    print(f"Matrix shape: {mat.shape}")
    
    if sum(score_list[:k]) > 0 and sum(doc_len) > 0:
        _p_w_q = np.dot(np.array([score_list[:k] / (doc_len + 1e-10), ]), mat)
        p_w_q = np.asarray(_p_w_q / sum(score_list[:k])).squeeze()
        rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]), order='token_scores')[::-1]
        
        print(f"\nRM1 computed. Terms: {len(rm1)}")
        print(f"Top 10 RM1 terms:")
        for i in range(min(10, len(rm1))):
            print(f"  {rm1[i]['tokens']}: {rm1[i]['token_scores']}")
        
        # Compute Clarity
        term_num = 100
        rm1_cut = rm1[:min(term_num, len(rm1))]
        p_w_q_norm = rm1_cut['token_scores'] / (rm1_cut['token_scores'].sum() + 1e-10)
        
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
        clarity_score = np.sum(p_w_q_norm * np.log2((p_w_q_norm + 1e-10) / (p_t_D + 1e-10)))
        
        print(f"\n{'='*80}")
        print(f"CLARITY SCORE: {clarity_score}")
        print(f"{'='*80}")
