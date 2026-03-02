#!/usr/bin/env python3
"""
Batch convert all retrieval results to ragnarok format.
Loads pyserini index ONCE and reuses it for all files - much faster!
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent

# Set up Java environment before importing pyserini
def setup_java_environment():
    """Set up Java environment variables needed for pyserini."""
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    jvm_path = "/future/u/negara/miniconda3/lib/server/libjvm.so"
    if os.path.exists(jvm_path):
        os.environ["JVM_PATH"] = jvm_path
        print(f"✅ Set JVM_PATH to: {jvm_path}")
    
    print(f"✅ Set JAVA_HOME to: {java_home}")

setup_java_environment()

try:
    from pyserini.search.lucene import LuceneSearcher
    print("✅ Pyserini imported successfully!")
except ImportError:
    print("❌ ERROR: Pyserini not found!")
    sys.exit(1)

def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from TSV file."""
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                qid, text = parts
                queries[qid] = text
    return queries

def load_results(results_file: str, k: int = 5) -> Dict[str, List[Dict]]:
    """Load TREC-style results and keep only top k per query."""
    query_results = {}
    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docid, rank, score = parts[:5]
                if qid not in query_results:
                    query_results[qid] = []
                query_results[qid].append({
                    'docid': docid,
                    'score': float(score),
                    'rank': int(rank)
                })
    
    # Keep only top k results per query
    for qid in query_results:
        query_results[qid] = sorted(query_results[qid], key=lambda x: x['rank'])[:k]
    
    return query_results

def convert_to_ragnarok(queries: Dict[str, str], 
                       query_results: Dict[str, List[Dict]], 
                       searcher: LuceneSearcher) -> List[Dict]:
    """Convert to ragnarok format using provided searcher."""
    ragnarok_format = []
    
    for qid, results in query_results.items():
        if qid not in queries:
            continue
        
        candidates = []
        for result in results:
            # Get document text
            try:
                base_docid = result['docid']
                doc = searcher.doc(base_docid)
                if doc and doc.raw():
                    doc_data = json.loads(doc.raw())
                    doc_text = doc_data.get('title', '') + " " + doc_data.get('segment', '')
                    if not doc_text.strip():
                        doc_text = f"Document {base_docid} - No text available"
                else:
                    doc_text = f"Document {base_docid} - Could not retrieve"
            except Exception as e:
                doc_text = f"Document {base_docid} - Error retrieving"
            
            candidate = {
                "doc": {"segment": doc_text},
                "docid": result['docid'],
                "score": result['score']
            }
            candidates.append(candidate)
        
        ragnarok_query = {
            "query": {
                "text": queries[qid],
                "qid": qid
            },
            "candidates": candidates
        }
        ragnarok_format.append(ragnarok_query)
    
    return ragnarok_format

def get_query_file_for_run(run_file: Path, base_dir: Path) -> Path:
    """Get the corresponding query file for a run file.
    
    Pattern: run.{name}.txt -> queries/topics.{name}.txt
    """
    # Extract the name part from run.{name}.txt
    run_name = run_file.stem.replace("run.", "")
    query_file = base_dir / "querygym/queries" / f"topics.{run_name}.txt"
    
    # Fallback to original queries if specific file doesn't exist
    if not query_file.exists():
        query_file = base_dir / "data/topics.rag24.test.txt"
    
    return query_file

def main():
    # Configuration
    BASE_DIR = _REPO
    
    INPUT_DIRS = [
        BASE_DIR / "querygym/retrieval_cohere",
        BASE_DIR / "querygym/retrieval"
    ]
    
    OUTPUT_DIRS = [
        BASE_DIR / "querygym/rag_prepared/retrieval_cohere",
        BASE_DIR / "querygym/rag_prepared/retrieval"
    ]
    
    K = 5  # Top k results per query
    
    print("=" * 60)
    print("Batch Converting All Retrieval Results to Ragnarok Format")
    print("=" * 60)
    print(f"📊 Converting top {K} results per query")
    print(f"📝 Using reformulated queries for each run file")
    print("")
    
    # Initialize pyserini searcher once
    print("🔍 Initializing Pyserini searcher for msmarco-v2.1-doc index...")
    print("⚠️  This may take several minutes (but only once!)...")
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented')
        print("✅ Pyserini searcher initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize searcher: {e}")
        sys.exit(1)
    
    # Process all directories
    total_files_processed = 0
    total_files_skipped = 0
    
    for input_dir, output_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
        dir_name = input_dir.name
        print("=" * 60)
        print(f"Processing: {dir_name}")
        print("=" * 60)
        
        # Get all run files
        run_files = sorted(input_dir.glob("run.*.txt"))
        print(f"Found {len(run_files)} run files\n")
        
        for i, run_file in enumerate(run_files, 1):
            output_file = output_dir / f"ragnarok_format_{run_file.stem}.json"
            
            # Skip if already exists
            if output_file.exists():
                print(f"[{i}/{len(run_files)}] ⏭️  Skipping (exists): {run_file.name}")
                total_files_skipped += 1
                continue
            
            print(f"[{i}/{len(run_files)}] 🔄 Processing: {run_file.name}")
            
            try:
                # Get the corresponding query file for this run
                query_file = get_query_file_for_run(run_file, BASE_DIR)
                print(f"  📖 Using queries from: {query_file.name}")
                
                # Load queries for this specific run
                queries = load_queries(str(query_file))
                print(f"  ✅ Loaded {len(queries)} queries")
                
                # Load results for this file
                query_results = load_results(str(run_file), k=K)
                print(f"  📊 Loaded {len(query_results)} queries with results")
                
                # Convert to ragnarok format (reusing the same searcher!)
                ragnarok_data = convert_to_ragnarok(queries, query_results, searcher)
                print(f"  ✅ Converted {len(ragnarok_data)} queries")
                
                # Save output
                with open(output_file, 'w') as f:
                    json.dump(ragnarok_data, f, indent=2)
                print(f"  💾 Saved to: {output_file.name}\n")
                
                total_files_processed += 1
                
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                continue
    
    # Summary
    print("=" * 60)
    print("🎉 Batch Conversion Complete!")
    print("=" * 60)
    print(f"✅ Files processed: {total_files_processed}")
    print(f"⏭️  Files skipped (already exist): {total_files_skipped}")
    print(f"📁 Output directories:")
    for output_dir in OUTPUT_DIRS:
        count = len(list(output_dir.glob("ragnarok_format_*.json")))
        print(f"   - {output_dir.name}: {count} files")
    print("")

if __name__ == "__main__":
    main()

