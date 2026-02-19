#!/usr/bin/env python3
"""
Script to run query reformulations through Pyserini with proper Java setup.
This script sets up Java environment variables before importing Pyserini.
"""

import os
import sys
from pathlib import Path

def setup_java_environment():
    """Set up Java environment variables for Pyserini."""
    print("🔧 Setting up Java environment for Pyserini...")
    
    # Set JAVA_HOME
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    # Try different possible JVM paths
    jvm_paths = [
        "/future/u/negara/miniconda3/lib/server/libjvm.so",
        "/future/u/negara/miniconda3/pkgs/openjdk-21.0.6-h38aa4c6_0/lib/server/libjvm.so",
        "/future/u/negara/miniconda3/envs/pyserini-env/lib/server/libjvm.so",
        "/future/u/negara/miniconda3/envs/py311/lib/server/libjvm.so",
    ]
    
    jvm_path = None
    for path in jvm_paths:
        if Path(path).exists():
            jvm_path = path
            break
    
    if jvm_path is None:
        print("❌ Could not find libjvm.so. Available options:")
        for path in jvm_paths:
            status = "✅" if Path(path).exists() else "❌"
            print(f"   {status} {path}")
        sys.exit(1)
    
    os.environ["JVM_PATH"] = jvm_path
    
    print(f"✅ JAVA_HOME: {java_home}")
    print(f"✅ JVM_PATH: {jvm_path}")
    
    return True

def main():
    """Main function with Java setup."""
    # Set up Java environment BEFORE importing Pyserini
    setup_java_environment()
    
    # Now import and run the main script functionality
    print("📦 Importing Pyserini...")
    
    try:
        import json
        import time
        from pyserini.search.lucene import LuceneSearcher
        print("✅ Pyserini imported successfully!")
    except Exception as e:
        print(f"❌ Failed to import Pyserini: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're in the pyserini-env conda environment")
        print("2. Install Pyserini: conda install -c conda-forge pyserini")
        print("3. Check Java installation: java -version")
        sys.exit(1)
    
    print("🚀 Starting Pyserini retrieval process...")
    
    # Initialize Pyserini searcher
    print("📥 Loading msmarco-v2.1-doc index...")
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segemented')
        print(f"✅ Index loaded successfully! Documents: {searcher.num_docs:,}")
    except Exception as e:
        print(f"❌ Error loading Pyserini index: {e}")
        print("The msmarco-v2.1-doc index will be downloaded automatically on first use.")
        print("This may take some time depending on your internet connection.")
        return
    
    # Define paths
    queries_dir = Path("/future/u/negara/home/RAG-Query/query_reformulation/queries")
    output_dir = Path("/future/u/negara/home/RAG-Query/query_reformulation/run")
    output_dir.mkdir(exist_ok=True)
    
    # Define the functions inline to ensure we have the updated score handling
    def read_query_file(file_path):
        """Read queries from a file and return list of (query_id, query_text) tuples."""
        queries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        query_id, query_text = parts
                        queries.append((query_id, query_text))
        return queries

    def search_pyserini_index(searcher, query_text, top_k=1000):
        """Search the Pyserini index for a given query."""
        try:
            hits = searcher.search(query_text, k=top_k)
            return hits
        except Exception as e:
            print(f"Error searching for query '{query_text}': {e}")
            return []

    def format_trec_results(query_id, hits, run_name="pyserini"):
        """Format search results in TREC format with real scores."""
        trec_lines = []
        for rank, hit in enumerate(hits, 1):
            try:
                # Pyserini hit structure - always has docid and score
                doc_id = hit.docid
                
                # Get the real score from the index
                if hasattr(hit, 'score'):
                    score = float(hit.score)  # Ensure it's a float
                else:
                    print(f"Warning: No score available for query {query_id}, rank {rank}, using 0.0")
                    score = 0.0  # Use 0.0 instead of rank-based fallback
                
                # TREC format: query_id Q0 doc_id rank score run_name
                trec_line = f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}"
                trec_lines.append(trec_line)
                
            except Exception as e:
                print(f"Error formatting result for query {query_id}, rank {rank}: {e}")
                continue
        
        return trec_lines

    def process_query_variant(searcher, input_file, output_file, variant_name):
        """Process a single query variant file."""
        print(f"Processing {variant_name}...")
        
        # Read queries
        queries = read_query_file(input_file)
        print(f"  Found {len(queries)} queries")
        
        # Process each query
        all_results = []
        for i, (query_id, query_text) in enumerate(queries):
            if i % 50 == 0:  # Progress update every 50 queries
                print(f"  Processing query {i+1}/{len(queries)}: {query_id}")
            
            # Search the index
            hits = search_pyserini_index(searcher, query_text)
            
            # Format results in TREC format
            trec_results = format_trec_results(query_id, hits, f"pyserini_{variant_name}")
            all_results.extend(trec_results)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.05)  # Shorter delay for Pyserini
        
        # Write results to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in all_results:
                f.write(line + '\n')
        
        print(f"  Saved {len(all_results)} results to {output_file}")
    
    # Process each variant file
    variant_files = sorted(queries_dir.glob("topics.rag24.test.v*.txt"))
    print(f"📂 Found {len(variant_files)} variant files to process")
    
    for i, variant_file in enumerate(variant_files, 1):
        variant_name = variant_file.stem.split('.')[-1]
        output_file = output_dir / f"run.rag24.pyserini.{variant_name}.txt"
        
        print(f"\n[{i}/{len(variant_files)}] Processing {variant_name}...")
        process_query_variant(searcher, variant_file, output_file, variant_name)
    
    print(f"\n🎉 Pyserini retrieval process completed!")
    print(f"📊 Processed {len(variant_files)} query variants")
    print(f"💾 Results saved in: {output_dir}")
    
    # Summary
    print(f"\n📋 Generated files:")
    for variant_file in variant_files:
        variant_name = variant_file.stem.split('.')[-1]
        output_file = output_dir / f"run.rag24.pyserini.{variant_name}.txt"
        if output_file.exists():
            print(f"  ✅ {output_file.name}")
        else:
            print(f"  ❌ {output_file.name} (failed)")

if __name__ == "__main__":
    main()
