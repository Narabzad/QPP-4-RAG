#!/usr/bin/env python3
"""
Retrieve documents for all query files using pyserini BM25.
Processes all 31 query files (6 methods × 5 trials + 1 original).
"""

import os
import sys
from pathlib import Path
import time
import argparse

def setup_java_environment():
    """Set up Java environment for Pyserini."""
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    # Try different possible JVM paths
    jvm_paths = [
        f"{java_home}/lib/server/libjvm.so",
        f"{java_home}/pkgs/openjdk-21.0.6-h38aa4c6_0/lib/server/libjvm.so",
        f"{java_home}/envs/pyserini-env/lib/server/libjvm.so",
        f"{java_home}/envs/py311/lib/server/libjvm.so",
    ]
    
    jvm_path = None
    for path in jvm_paths:
        if Path(path).exists():
            jvm_path = path
            break
    
    if jvm_path:
        os.environ["JVM_PATH"] = jvm_path
        print(f"✅ Set JVM_PATH to: {jvm_path}")
    else:
        print("⚠️  Warning: Could not find libjvm.so. Pyserini may not work.")
    
    print(f"✅ Set JAVA_HOME to: {java_home}")

def read_topics_file(topics_file):
    """Read queries from a topics file."""
    queries = []
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    query_id, query_text = parts
                    queries.append((query_id, query_text))
                else:
                    print(f"⚠️  Invalid line {line_num}: {line}")
    
    return queries

def search_pyserini_index(searcher, query_text, k=100):
    """Search the Pyserini index for the given query."""
    try:
        hits = searcher.search(query_text, k=k)
        return hits
    except Exception as e:
        print(f"❌ Error searching for query: {e}")
        return []

def format_trec_results(query_id, hits, run_name="pyserini"):
    """Format search results in TREC format."""
    trec_lines = []
    
    for i, hit in enumerate(hits):
        if hit.docid and hit.score:
            trec_lines.append(f"{query_id} Q0 {hit.docid} {i+1} {hit.score:.6f} {run_name}")
        else:
            continue
    
    return trec_lines

def process_query_file(searcher, topics_file, output_dir, k=100):
    """Process a single query file and save results."""
    
    # Extract run name from filename
    # e.g., "topics.genqr_trial1.txt" -> "genqr_trial1"
    # or "topics.original.txt" -> "original"
    filename = topics_file.name
    if filename.startswith("topics.") and filename.endswith(".txt"):
        run_name = filename[7:-4]  # Remove "topics." prefix and ".txt" suffix
    else:
        run_name = topics_file.stem
    
    print(f"\n📄 Processing: {filename}")
    print(f"   Run name: {run_name}")
    
    # Read queries
    queries = read_topics_file(topics_file)
    print(f"   Found {len(queries)} queries")
    
    if not queries:
        print(f"   ⚠️  No queries found, skipping...")
        return False
    
    # Create output file
    output_file = output_dir / f"run.{run_name}.txt"
    
    # Process each query
    all_results = []
    start_time = time.time()
    
    for i, (query_id, query_text) in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"   Processing query {i+1}/{len(queries)}: {query_id}")
        
        # Search the index
        hits = search_pyserini_index(searcher, query_text, k)
        
        # Format results in TREC format
        trec_results = format_trec_results(query_id, hits, run_name)
        all_results.extend(trec_results)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.01)
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(line + '\n')
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"   ✅ Saved {len(all_results)} result lines to {output_file.name}")
    print(f"   ⏱️  Time: {duration:.2f} seconds")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Retrieve documents for all query files using pyserini BM25')
    parser.add_argument('--queries-dir', type=str, 
                       default='/future/u/negara/home/set_based_QPP/querygym/queries',
                       help='Directory containing query files (default: querygym/queries)')
    parser.add_argument('--output-dir', type=str,
                       default='/future/u/negara/home/set_based_QPP/querygym/retrieval',
                       help='Output directory for retrieval results (default: querygym/retrieval)')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of documents to retrieve per query (default: 100)')
    
    args = parser.parse_args()
    
    # Set up Java environment BEFORE importing Pyserini
    print("🔧 Setting up Java environment...")
    setup_java_environment()
    
    # Now import Pyserini
    print("📦 Importing Pyserini...")
    try:
        from pyserini.search.lucene import LuceneSearcher
        print("✅ Pyserini imported successfully!")
    except Exception as e:
        print(f"❌ Failed to import Pyserini: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're in the pyserini-env conda environment")
        print("2. Install Pyserini: conda install -c conda-forge pyserini")
        print("3. Check Java installation: java -version")
        sys.exit(1)
    
    # Initialize Pyserini searcher
    print("\n📥 Loading msmarco-v2.1-doc-segmented index...")
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented')
        print(f"✅ Index loaded successfully! Documents: {searcher.num_docs:,}")
    except Exception as e:
        print(f"❌ Error loading Pyserini index: {e}")
        print("The msmarco-v2.1-doc-segmented index will be downloaded automatically on first use.")
        print("This may take some time depending on your internet connection.")
        return
    
    # Find all query files
    queries_dir = Path(args.queries_dir)
    if not queries_dir.exists():
        print(f"❌ Queries directory not found: {queries_dir}")
        print("Please run generate_query_files.py first to generate query files.")
        return
    
    query_files = sorted(queries_dir.glob("topics.*.txt"))
    
    if not query_files:
        print(f"❌ No query files found in {queries_dir}")
        print("Please run generate_query_files.py first to generate query files.")
        return
    
    print(f"\n📋 Found {len(query_files)} query files to process")
    print(f"🔍 K={args.k} documents per query")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Results will be saved to: {output_dir}")
    
    # Process each query file
    print("\n" + "=" * 60)
    print("🚀 Starting retrieval process...")
    print("=" * 60)
    
    successful = []
    failed = []
    overall_start = time.time()
    
    for i, query_file in enumerate(query_files, 1):
        print(f"\n[{i}/{len(query_files)}] Processing {query_file.name}...")
        
        try:
            if process_query_file(searcher, query_file, output_dir, args.k):
                successful.append(query_file.name)
            else:
                failed.append(query_file.name)
        except Exception as e:
            print(f"   ❌ Error processing {query_file.name}: {e}")
            failed.append(query_file.name)
    
    overall_end = time.time()
    overall_duration = overall_end - overall_start
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 RETRIEVAL PROCESS COMPLETED!")
    print("=" * 60)
    print(f"✅ Successful: {len(successful)}/{len(query_files)}")
    print(f"❌ Failed: {len(failed)}/{len(query_files)}")
    print(f"⏱️  Total time: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes)")
    print(f"💾 Results saved to: {output_dir}")
    
    if failed:
        print(f"\n⚠️  Failed files:")
        for f in failed:
            print(f"   - {f}")

if __name__ == "__main__":
    main()







