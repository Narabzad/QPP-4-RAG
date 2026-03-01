#!/usr/bin/env python3
"""
Retrieve documents for all query files using Cohere DiskVectorIndex for semantic search.
Uses Cohere's pre-built index for MS MARCO v2.1.
"""

import os
import sys
from pathlib import Path
import time
import argparse


def get_cohere_api_key():
    """Get Cohere API key from environment variable or .env.local file."""
    # Try environment variable first
    api_key = os.environ.get('COHERE_API_KEY') or os.environ.get('CO_API_KEY')
    
    if not api_key:
        # Try to load from .env.local file using dotenv
        try:
            from dotenv import load_dotenv
            # Try to find .env.local in common locations
            env_paths = [
                Path(__file__).resolve().parent.parent / '.env.local',  # Project root relative to script
                Path('.env.local'),
            ]
            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path))
                    api_key = os.getenv('CO_API_KEY') or os.getenv('COHERE_API_KEY')
                    if api_key:
                        break
        except ImportError:
            pass
    
    if not api_key:
        # Try to get from ragnarok's api_keys module if available
        try:
            from ragnarok.generate.api_keys import get_cohere_api_key
            api_key = get_cohere_api_key()
        except (ImportError, Exception):
            pass
    
    if not api_key:
        raise ValueError(
            "Cohere API key not found. Please set COHERE_API_KEY environment variable, "
            "or ensure .env.local file exists with CO_API_KEY set."
        )
    # Strip any whitespace and set as environment variable for DiskVectorIndex
    api_key = api_key.strip()
    os.environ['COHERE_API_KEY'] = api_key
    return api_key

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

def search_with_cohere_index(index, query_text, k=100):
    """Search using Cohere DiskVectorIndex."""
    try:
        docs = index.search(query_text, top_k=k)
        return docs
    except Exception as e:
        print(f"❌ Error searching with Cohere index: {e}")
        return []

def format_trec_results(query_id, docs, run_name="cohere"):
    """Format Cohere search results in TREC format (matching pyserini output exactly)."""
    trec_lines = []
    
    for rank, doc in enumerate(docs, 1):
        # Cohere DiskVectorIndex returns documents in nested structure:
        # doc['doc']['docid'] - document ID
        # doc['score'] - relevance score
        doc_id = None
        score = None
        
        if isinstance(doc, dict):
            # Check nested structure first (DiskVectorIndex format)
            if 'doc' in doc and isinstance(doc['doc'], dict):
                doc_id = doc['doc'].get('docid', doc['doc'].get('id', ''))
                score = doc.get('score', None)
            else:
                # Try flat structure
                doc_id = doc.get('id', doc.get('docid', doc.get('doc_id', '')))
                score = doc.get('score', doc.get('relevance_score', doc.get('similarity')))
        elif hasattr(doc, 'id') or hasattr(doc, 'docid'):
            doc_id = getattr(doc, 'id', getattr(doc, 'docid', ''))
            score = getattr(doc, 'score', getattr(doc, 'relevance_score', None))
        elif isinstance(doc, str):
            # If doc is just a string (doc_id), use it directly
            doc_id = doc
            score = None
        elif isinstance(doc, (list, tuple)) and len(doc) > 0:
            # If doc is a list/tuple, first element is doc_id
            doc_id = str(doc[0])
            score = doc[1] if len(doc) > 1 else None
        else:
            # Try to convert to string
            doc_id = str(doc)
            score = None
        
        if doc_id:
            # If no score provided, use a default decreasing score (similar to BM25)
            if score is None:
                score = 1000.0 - (rank - 1)
            
            # Format: query_id Q0 doc_id rank score run_name (matching pyserini format exactly)
            trec_lines.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}")
    
    return trec_lines

def process_query_file(cohere_index, topics_file, output_dir, k=100):
    """Process a single query file and save results."""
    
    # Extract run name from filename
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
        
        # Search using Cohere DiskVectorIndex
        docs = search_with_cohere_index(cohere_index, query_text, k)
        
        if not docs:
            print(f"   ⚠️  No results for query {query_id}, skipping...")
            continue
        
        # Format results in TREC format
        trec_results = format_trec_results(query_id, docs, run_name)
        all_results.extend(trec_results)
        
        # Small delay to avoid overwhelming the API
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
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Retrieve documents for all query files using Cohere DiskVectorIndex')
    parser.add_argument('--queries-dir', type=str,
                       default=str(_here / "queries"),
                       help='Directory containing query files (default: querygym/queries)')
    parser.add_argument('--output-dir', type=str,
                       default=str(_here / "retrieval_cohere"),
                       help='Output directory for retrieval results (default: querygym/retrieval_cohere)')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of documents to retrieve per query (default: 100)')
    parser.add_argument('--candidate-k', type=int, default=1000,
                       help='Number of candidate documents to fetch before reranking (default: 1000)')
    parser.add_argument('--index-name', type=str, default='Cohere/trec-rag-2024-index',
                       help='Cohere DiskVectorIndex name (default: Cohere/trec-rag-2024-index)')
    parser.add_argument('--test-file', type=str, default=None,
                       help='Test with a single query file (e.g., topics.original.txt)')
    
    args = parser.parse_args()
    
    # Get Cohere API key
    print("🔑 Getting Cohere API key...")
    try:
        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            key_preview = f"{cohere_api_key[:4]}...{cohere_api_key[-4:]}" if len(cohere_api_key) > 8 else "***"
            print(f"✅ Cohere API key found! (Key: {key_preview})")
        else:
            raise ValueError("API key is empty")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Initialize Cohere DiskVectorIndex
    print("📦 Importing DiskVectorIndex...")
    try:
        from DiskVectorIndex import DiskVectorIndex
        print("✅ DiskVectorIndex imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import DiskVectorIndex: {e}")
        print("\nTroubleshooting tips:")
        print("1. Install DiskVectorIndex: pip install DiskVectorIndex")
        print("2. Ensure COHERE_API_KEY environment variable is set")
        sys.exit(1)
    
    # Load Cohere index
    print(f"\n📥 Loading Cohere index: {args.index_name}...")
    try:
        cohere_index = DiskVectorIndex(args.index_name)
        print(f"✅ Cohere index loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Cohere index: {e}")
        print(f"\nTroubleshooting tips:")
        print("1. Ensure the index name is correct: {args.index_name}")
        print("2. Check that COHERE_API_KEY is set correctly")
        print("3. The index will be downloaded automatically on first use")
        sys.exit(1)
    
    # Find all query files
    queries_dir = Path(args.queries_dir)
    if not queries_dir.exists():
        print(f"❌ Queries directory not found: {queries_dir}")
        return
    
    query_files = sorted(queries_dir.glob("topics.*.txt"))
    
    if not query_files:
        print(f"❌ No query files found in {queries_dir}")
        return
    
    # If test file specified, only process that one
    if args.test_file:
        test_path = queries_dir / args.test_file
        if test_path.exists():
            query_files = [test_path]
            print(f"\n🧪 TEST MODE: Processing only {args.test_file}")
        else:
            print(f"❌ Test file not found: {test_path}")
            return
    
    print(f"\n📋 Found {len(query_files)} query files to process")
    print(f"🔍 Retrieval: Cohere DiskVectorIndex semantic search → top-{args.k}")
    print(f"📚 Index: {args.index_name}")
    
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
            if process_query_file(cohere_index, query_file, output_dir, args.k):
                successful.append(query_file.name)
            else:
                failed.append(query_file.name)
        except Exception as e:
            print(f"   ❌ Error processing {query_file.name}: {e}")
            import traceback
            traceback.print_exc()
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

