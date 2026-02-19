#!/usr/bin/env python3
"""
Compute Reciprocal Volume (RV) scores as a pre-retrieval QPP method.
Uses both BERT and E5 embeddings to compute RV scores for each query
based on its reformulations.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

# Try to import required libraries
try:
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ ERROR: Required packages not found: {e}")
    print("   Please install: transformers, sentence-transformers")
    sys.exit(1)

def reciprocal_volume(query_embedding, doc_embeddings):
    """
    Computes the Reciprocal Volume (RV) for a given query and document embeddings.
    
    Parameters:
    - query_embedding: np.ndarray of shape (d,)
        Dense embedding of the query.
    - doc_embeddings: np.ndarray of shape (k, d)
        Dense embeddings of the documents (reformulations).
    
    Returns:
    - rv_score: float
        Reciprocal Volume score.
    """
    if doc_embeddings.shape[0] == 0:
        return None
    
    # Stack query and document embeddings to compute the hypercube
    combined_embeddings = np.vstack([query_embedding, doc_embeddings])  # Shape: (k+1, d)
    
    # Calculate the side length for each dimension
    max_vals = np.max(combined_embeddings, axis=0)
    min_vals = np.min(combined_embeddings, axis=0)
    side_lengths = max_vals - min_vals  # Shape: (d,)
    
    # Avoid log(0) by adding small epsilon
    side_lengths = np.maximum(side_lengths, 1e-10)
    
    # Compute the log of side lengths
    log_side_lengths = np.log(side_lengths)
    
    # Compute the Reciprocal Volume
    sum_log = np.sum(log_side_lengths)
    if sum_log == 0 or np.isinf(sum_log) or np.isnan(sum_log):
        return None
    
    rv_score = -1 / sum_log
    
    return rv_score


def generate_bert_embeddings(texts, tokenizer, model, device='cpu', batch_size=32):
    """Generate embeddings using BERT model."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of the last hidden state outputs as embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def generate_e5_embeddings(texts, model, device='cpu', batch_size=32):
    """Generate embeddings using E5 model."""
    # E5 models expect specific prefixes, but for general use we can use without
    # For E5-base, we can use the model directly
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # E5 models are typically used with SentenceTransformer interface
        batch_embeddings = model.encode(batch_texts, batch_size=len(batch_texts), 
                                       convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def load_queries(queries_file):
    """Load queries from topics file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id] = query_text
    return queries


def process_all_queries(base_dir, output_file):
    """Process all queries and compute RV scores for each reformulation."""
    
    print("🚀 Computing RV scores for QueryGym queries...")
    print("="*80)
    
    # Initialize models
    print("\n📥 Loading BERT model...")
    device = 'cpu'  # Use CPU to avoid GPU memory issues
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_model.eval()
    print("✅ BERT model loaded!")
    
    print("\n📥 Loading E5 model...")
    try:
        # Try E5-base-v2 first, then E5-base
        e5_model = SentenceTransformer('intfloat/e5-base-v2', device=device)
        print("✅ E5-base-v2 model loaded!")
    except:
        try:
            e5_model = SentenceTransformer('intfloat/e5-base', device=device)
            print("✅ E5-base model loaded!")
        except Exception as e:
            print(f"⚠️  Could not load E5 model: {e}")
            print("   Trying alternative: sentence-transformers/all-MiniLM-L6-v2")
            e5_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
            print("✅ Alternative model loaded!")
    
    # Load all queries and reformulations with method/trial info
    queries_dir = Path(base_dir) / "queries"
    original_queries = load_queries(queries_dir / "topics.original.txt")
    
    # Collect all reformulations for each query with method/trial info
    query_reformulations = defaultdict(list)  # qid -> list of (method_trial, text)
    
    methods = ['genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    for method in methods:
        for trial in range(1, 6):
            query_file = queries_dir / f"topics.{method}_trial{trial}.txt"
            if query_file.exists():
                reform_queries = load_queries(query_file)
                for qid, qtext in reform_queries.items():
                    method_key = f"{method}_{trial}"
                    query_reformulations[qid].append((method_key, qtext))
    
    # Also add original
    for qid, qtext in original_queries.items():
        query_reformulations[qid].insert(0, ('original_None', qtext))
    
    print(f"\n📊 Found {len(original_queries)} queries")
    print(f"📊 Average reformulations per query: {np.mean([len(v) for v in query_reformulations.values()]):.1f}")
    
    # Compute RV scores for each reformulation
    # Structure: results[qid][method_trial] = {'rv_bert': ..., 'rv_e5': ...}
    results = defaultdict(dict)
    
    print("\n🔄 Computing RV scores for each reformulation...")
    processed = 0
    
    for qid, reformulations in query_reformulations.items():
        if len(reformulations) < 2:  # Need at least 2 texts (query + 1 reformulation)
            continue
        
        # Extract all texts
        all_texts = [text for _, text in reformulations]
        method_keys = [key for key, _ in reformulations]
        
        try:
            # BERT embeddings
            bert_embeddings = generate_bert_embeddings(all_texts, bert_tokenizer, bert_model, device=device)
            
            # E5 embeddings
            e5_embeddings = generate_e5_embeddings(all_texts, e5_model, device=device)
            
            # For each reformulation, compute RV using it as query and others as documents
            for i, (method_key, query_text) in enumerate(reformulations):
                # Use this reformulation as the "query"
                query_bert_emb = bert_embeddings[i]
                query_e5_emb = e5_embeddings[i]
                
                # Use all other reformulations as "documents"
                other_indices = [j for j in range(len(reformulations)) if j != i]
                if len(other_indices) == 0:
                    continue
                
                other_bert_embs = bert_embeddings[other_indices]
                other_e5_embs = e5_embeddings[other_indices]
                
                # Compute RV scores
                rv_bert = reciprocal_volume(query_bert_emb, other_bert_embs)
                rv_e5 = reciprocal_volume(query_e5_emb, other_e5_embs)
                
                results[qid][method_key] = {
                    'rv_bert': float(rv_bert) if rv_bert is not None else None,
                    'rv_e5': float(rv_e5) if rv_e5 is not None else None
                }
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{len(original_queries)} queries...")
                
        except Exception as e:
            print(f"  ⚠️  Error processing query {qid}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Processed {processed} queries")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Also create CSV
    csv_file = output_path.with_suffix('.csv')
    import pandas as pd
    rows = []
    for qid, reform_scores in results.items():
        for method_key, scores in reform_scores.items():
            rows.append({
                'query_id': qid,
                'method': method_key,
                'rv_bert': scores['rv_bert'],
                'rv_e5': scores['rv_e5']
            })
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"💾 CSV saved to: {csv_file}")
    
    return results


def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    output_file = base_dir / "qpp" / "rv_scores.json"
    
    results = process_all_queries(base_dir, output_file)
    
    # Count total reformulations with scores
    total_reforms = sum(len(reform_scores) for reform_scores in results.values())
    total_with_bert = sum(sum(1 for s in reform_scores.values() if s['rv_bert'] is not None) 
                         for reform_scores in results.values())
    total_with_e5 = sum(sum(1 for s in reform_scores.values() if s['rv_e5'] is not None) 
                       for reform_scores in results.values())
    
    print("\n🎉 Done!")
    print(f"📊 Total reformulations: {total_reforms}")
    print(f"📊 Reformulations with RV_BERT scores: {total_with_bert}")
    print(f"📊 Reformulations with RV_E5 scores: {total_with_e5}")


if __name__ == "__main__":
    main()
