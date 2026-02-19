#!/usr/bin/env python3
"""
Run BERT-QPP (both bi-encoder and cross-encoder) on querygym queries.
For each query/reformulation, uses the first retrieved document.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

# Setup Java environment for Pyserini
def setup_java_environment():
    """Set up Java environment variables needed for pyserini."""
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    jvm_path = "/future/u/negara/miniconda3/lib/server/libjvm.so"
    if os.path.exists(jvm_path):
        os.environ["JVM_PATH"] = jvm_path

setup_java_environment()

try:
    from pyserini.search.lucene import LuceneSearcher
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers import SentenceTransformer, util
    import torch
except ImportError as e:
    print(f"❌ ERROR: Required packages not found: {e}")
    print("   Please install: sentence-transformers, pyserini")
    sys.exit(1)

class DocumentRetriever:
    """Class to retrieve document text from TREC RAG index."""
    
    def __init__(self, index_name='msmarco-v2.1-doc-segmented'):
        """Initialize the document retriever."""
        print(f"🔍 Initializing Pyserini searcher for {index_name}...")
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        print("✅ Searcher initialized!\n")
    
    def get_document_text(self, docid: str) -> str:
        """Retrieve the text of a document given its ID."""
        try:
            doc = self.searcher.doc(docid)
            if doc and doc.raw():
                doc_data = json.loads(doc.raw())
                title = doc_data.get('title', '')
                segment = doc_data.get('segment', '')
                full_text = title + " " + segment if title else segment
                return full_text.strip()
            return ""
        except Exception as e:
            print(f"⚠️  Error retrieving {docid}: {e}")
            return ""

def load_retrieval_run_file(run_file_path):
    """Load TREC format run file and return first doc per query."""
    first_docs = {}
    with open(run_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid = parts[0]
                docid = parts[2]
                if qid not in first_docs:
                    first_docs[qid] = docid
    return first_docs

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

# Global model instances to avoid reloading
_ce_model = None
_bi_model = None

def get_cross_encoder_model(model_path):
    """Get or create cross-encoder model instance."""
    global _ce_model
    if _ce_model is None:
        print(f"📥 Loading cross-encoder model (this may take a moment)...")
        _ce_model = CrossEncoder(model_path, num_labels=1, device='cpu')  # Use CPU to avoid GPU memory issues
        print(f"✅ Cross-encoder model loaded!")
    return _ce_model

def get_bi_encoder_model(model_path):
    """Get or create bi-encoder model instance."""
    global _bi_model
    if _bi_model is None:
        print(f"📥 Loading bi-encoder model (this may take a moment)...")
        _bi_model = SentenceTransformer(model_path, device='cpu')  # Use CPU to avoid GPU memory issues
        print(f"✅ Bi-encoder model loaded!")
    return _bi_model

def run_bert_qpp_cross_encoder(model_path, query_text, doc_text):
    """Run BERT-QPP cross-encoder model."""
    try:
        model = get_cross_encoder_model(model_path)
        score = model.predict([[query_text, doc_text]])
        return float(score[0])
    except Exception as e:
        print(f"⚠️  Error in cross-encoder: {e}")
        return None

def run_bert_qpp_bi_encoder(model_path, query_text, doc_text):
    """Run BERT-QPP bi-encoder model."""
    try:
        model = get_bi_encoder_model(model_path)
        query_emb = model.encode(query_text, convert_to_tensor=False)  # Use numpy instead of tensor
        doc_emb = model.encode(doc_text, convert_to_tensor=False)
        # Compute cosine similarity manually
        import numpy as np
        cosine_score = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        return float(cosine_score)
    except Exception as e:
        print(f"⚠️  Error in bi-encoder: {e}")
        return None

def process_all_queries(base_dir, bert_qpp_dir):
    """Process all queries and reformulations."""
    
    # Initialize document retriever
    retriever = DocumentRetriever()
    
    # Load models - check both zip and extracted versions
    models_dir = bert_qpp_dir / "models"
    ce_model_zip = models_dir / "tuned_model-ce_bert-base-uncased_e1_b8-20260208T030512Z-1-001.zip"
    ce_model_dir = models_dir / "tuned_model-ce_bert-base-uncased_e1_b8" / "tuned_model-ce_bert-base-uncased_e1_b8"
    ce_model_dir_alt = models_dir / "tuned_model-ce_bert-base-uncased_e1_b8"
    bi_model_dir = models_dir / "tuned_model_bi_bert-base-uncased_e_1_b_8" / "tuned_model-bi_bert-base-uncased_e_1_b_8"
    bi_model_dir_alt = models_dir / "tuned_model_bi_bert-base-uncased_e_1_b_8"
    
    # Check if models exist
    use_cross_encoder = False
    ce_model_path = None
    
    # Check nested directory first (from zip extraction)
    if ce_model_dir.exists() and (ce_model_dir / "pytorch_model.bin").exists():
        ce_model_path = str(ce_model_dir)
        use_cross_encoder = True
    elif ce_model_dir_alt.exists() and (ce_model_dir_alt / "pytorch_model.bin").exists():
        ce_model_path = str(ce_model_dir_alt)
        use_cross_encoder = True
    elif ce_model_zip.exists():
        # Try to use zip directly (sentence-transformers might support it)
        ce_model_path = str(ce_model_zip)
        use_cross_encoder = True
    
    use_bi_encoder = False
    bi_model_path = None
    
    # Check nested directory first (from zip extraction)
    if bi_model_dir.exists():
        # Check if it has model files (config.json or similar)
        if (bi_model_dir / "config.json").exists() or (bi_model_dir / "modules.json").exists():
            bi_model_path = str(bi_model_dir)
            use_bi_encoder = True
    elif bi_model_dir_alt.exists():
        if (bi_model_dir_alt / "config.json").exists() or (bi_model_dir_alt / "modules.json").exists():
            bi_model_path = str(bi_model_dir_alt)
            use_bi_encoder = True
    
    if not use_cross_encoder and not use_bi_encoder:
        print("⚠️  No BERT-QPP models found!")
        print(f"   Checked: {ce_model_dir}")
        print(f"   Checked: {ce_model_zip}")
        print(f"   Checked: {bi_model_dir}")
        return {}
    
    if use_cross_encoder:
        print(f"✅ Found cross-encoder model: {ce_model_path}")
    if use_bi_encoder:
        print(f"✅ Found bi-encoder model: {bi_model_path}")
    
    # Get all query files and run files
    queries_dir = base_dir / "queries"
    retrieval_dir_pyserini = base_dir / "retrieval"
    retrieval_dir_cohere = base_dir / "retrieval_cohere"
    
    results = {}
    
    # Process each reformulation for both retrieval methods
    methods = ['original', 'genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    
    for retrieval_method in ['pyserini', 'cohere']:
        retrieval_dir = retrieval_dir_pyserini if retrieval_method == 'pyserini' else retrieval_dir_cohere
        print(f"\n{'='*80}")
        print(f"🔍 Processing {retrieval_method.upper()} retrieval runs...")
        print(f"{'='*80}")
        
        for method in methods:
            if method == 'original':
                query_file = queries_dir / "topics.original.txt"
                run_file = retrieval_dir / "run.original.txt"
                if query_file.exists() and run_file.exists():
                    print(f"\n📊 Processing original ({retrieval_method})...")
                    process_reformulation(query_file, run_file, retriever,
                                        ce_model_path if use_cross_encoder else None,
                                        bi_model_path if use_bi_encoder else None,
                                        results, "original", retrieval_method)
            else:
                for trial in range(1, 6):
                    query_file = queries_dir / f"topics.{method}_trial{trial}.txt"
                    run_file = retrieval_dir / f"run.{method}_trial{trial}.txt"
                    
                    if not query_file.exists() or not run_file.exists():
                        continue
                    
                    print(f"\n📊 Processing {method}_trial{trial} ({retrieval_method})...")
                    process_reformulation(query_file, run_file, retriever, 
                                        ce_model_path if use_cross_encoder else None,
                                        bi_model_path if use_bi_encoder else None,
                                        results, f"{method}_trial{trial}", retrieval_method)
    
    return results

def process_reformulation(query_file, run_file, retriever, ce_model_path, bi_model_path, results, method_name, retrieval_method):
    """Process a single reformulation for a specific retrieval method."""
    queries = load_queries(query_file)
    first_docs = load_retrieval_run_file(run_file)
    
    processed = 0
    for qid, query_text in queries.items():
        if qid not in first_docs:
            continue
        
        docid = first_docs[qid]
        doc_text = retriever.get_document_text(docid)
        
        if not doc_text:
            continue
        
        # Initialize result dict for this query/method/retrieval_method combination
        key = f"{qid}_{method_name}_{retrieval_method}"
        if key not in results:
            results[key] = {
                'qid': qid,
                'method': method_name,
                'retrieval_method': retrieval_method,
                'query': query_text,
                'docid': docid
            }
        
        # Run cross-encoder if available
        if ce_model_path:
            ce_score = run_bert_qpp_cross_encoder(ce_model_path, query_text, doc_text)
            if ce_score is not None:
                results[key]['bert_qpp_cross_encoder'] = ce_score
        
        # Run bi-encoder if available
        if bi_model_path:
            bi_score = run_bert_qpp_bi_encoder(bi_model_path, query_text, doc_text)
            if bi_score is not None:
                results[key]['bert_qpp_bi_encoder'] = bi_score
        
        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(queries)} queries...")
    
    print(f"  ✅ Completed {method_name} ({retrieval_method}): {processed} queries")

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    bert_qpp_dir = Path("/future/u/negara/home/set_based_QPP/BERTQPP")
    output_dir = base_dir / "bert_qpp_results"
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Running BERT-QPP on QueryGym queries...")
    print("="*80)
    
    results = process_all_queries(base_dir, bert_qpp_dir)
    
    # Save results
    output_file = output_dir / "bert_qpp_scores.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📊 Total queries processed: {len(results)}")
    
    # Also create per-query CSV
    csv_file = output_dir / "bert_qpp_scores.csv"
    import pandas as pd
    
    rows = []
    for key, data in results.items():
        row = {
            'qid': data['qid'],
            'method': data['method'],
            'retrieval_method': data.get('retrieval_method', 'pyserini'),
            'bert_qpp_cross_encoder': data.get('bert_qpp_cross_encoder'),
            'bert_qpp_bi_encoder': data.get('bert_qpp_bi_encoder')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"💾 CSV saved to: {csv_file}")

if __name__ == "__main__":
    main()
