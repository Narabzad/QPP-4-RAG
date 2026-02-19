#!/usr/bin/env python3
"""
Direct retrieval script for topics files.
This script retrieves documents for queries in topics format files.
"""

import os
import sys
import argparse
from pathlib import Path
import time

def setup_java_environment():
    """Set up Java environment for Pyserini."""
    # Set JAVA_HOME to the conda environment
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    os.environ["JVM_PATH"] = f"{java_home}/lib/server/libjvm.so"
    
    print("🔧 Setting up Java environment for Pyserini...")
    print(f"✅ JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    print(f"✅ JVM_PATH: {os.environ.get('JVM_PATH')}")

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

def search_pyserini_index(searcher, query_text, k=1000):
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

def main():
    parser = argparse.ArgumentParser(description='Retrieve documents for queries in topics format')
    parser.add_argument('--topics-file', type=str, required=True,
                       help='Path to topics file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for retrieval results')
    parser.add_argument('--run-name', type=str, default="topics_retrieval",
                       help='Name for the run (default: topics_retrieval)')
    parser.add_argument('--k', type=int, default=1000,
                       help='Number of documents to retrieve per query (default: 1000)')
    
    args = parser.parse_args()
    
    # Set up Java environment BEFORE importing Pyserini
    setup_java_environment()
    
    # Now import and run the main script functionality
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
    
    print("🚀 Starting Pyserini retrieval process...")
    
    # Initialize Pyserini searcher
    print("📥 Loading msmarco-v2.1-doc-segmented index...")
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented')
        
        if searcher is None:
            print("❌ Could not load msmarco-v2.1-doc-segmented index")
            return
        else:
            print(f"✅ Index loaded successfully! Documents: {searcher.num_docs:,}")
    except Exception as e:
        print(f"❌ Error loading Pyserini index: {e}")
        print("The msmarco-v2.1-doc-segmented index will be downloaded automatically on first use.")
        print("This may take some time depending on your internet connection.")
        return
    
    # Read topics file
    topics_file = Path(args.topics_file)
    if not topics_file.exists():
        print(f"❌ Topics file not found: {topics_file}")
        return
    
    print(f"📖 Reading topics from: {topics_file}")
    queries = read_topics_file(topics_file)
    print(f"📋 Found {len(queries)} queries")
    
    if not queries:
        print("❌ No queries found in topics file")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    output_file = output_dir / f"run.{args.run_name}.txt"
    
    print(f"🔍 Starting retrieval for {len(queries)} queries...")
    print(f"💾 Results will be saved to: {output_file}")
    
    # Process each query
    all_results = []
    start_time = time.time()
    
    for i, (query_id, query_text) in enumerate(queries):
        print(f"  Processing query {i+1}/{len(queries)}: {query_id} - {query_text[:50]}...")
        
        # Search the index
        hits = search_pyserini_index(searcher, query_text, args.k)
        
        # Format results in TREC format
        trec_results = format_trec_results(query_id, hits, args.run_name)
        all_results.extend(trec_results)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.05)
    
    # Write results to output file
    print(f"💾 Writing results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(line + '\n')
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎉 Pyserini retrieval process completed!")
    print(f"📊 Processed {len(queries)} queries")
    print(f"📄 Generated {len(all_results)} result lines")
    print(f"💾 Results saved to: {output_file}")
    print(f"⏱️  Total time: {duration:.2f} seconds")
    
    # Summary
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"✅ Output file: {output_file.name} ({file_size:,} bytes)")
    else:
        print(f"❌ Output file creation failed: {output_file}")

if __name__ == "__main__":
    main()
