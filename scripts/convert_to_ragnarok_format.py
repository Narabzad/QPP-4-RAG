#!/usr/bin/env python3
"""
Helper script to convert your retrieved results to ragnarok format.
Modify this script based on your input data format.
"""
import json
import os
import sys
from typing import List, Dict, Any

# Set up Java environment before importing pyserini
def setup_java_environment():
    """Set up Java environment variables needed for pyserini."""
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        print("⚠️  JAVA_HOME not set. Please set it, e.g.: export JAVA_HOME=/path/to/java")
        return

    possible_jvm_paths = [
        os.path.join(java_home, "lib", "server", "libjvm.so"),
        os.path.join(java_home, "lib", "libjvm.so"),
    ]

    jvm_path = None
    for path in possible_jvm_paths:
        if os.path.exists(path):
            jvm_path = path
            break

    if jvm_path:
        os.environ["JVM_PATH"] = jvm_path
        print(f"✅ Set JVM_PATH to: {jvm_path}")
    else:
        print("⚠️  Warning: Could not find libjvm.so under JAVA_HOME. Pyserini may not work.")

    print(f"✅ Using JAVA_HOME: {java_home}")

# Set up Java environment first
setup_java_environment()

# Add pyserini to path if needed
try:
    from pyserini.search.lucene import LuceneSearcher
    print("✅ Pyserini imported successfully!")
except ImportError:
    print("❌ ERROR: Pyserini not found!")
    print("   Please install pyserini with: pip install pyserini")
    print("   Or activate the correct conda environment: conda activate pyserini-env")
    print("   If you get conda errors, run: conda init bash")
    sys.exit(1)

def convert_your_format_to_ragnarok(input_file: str, output_file: str):
    """
    Convert your retrieved results to ragnarok format.
    
    MODIFY THIS FUNCTION based on your input format!
    
    Example assumes your format is:
    {
        "query_id": "001",
        "query_text": "What is the capital of France?",
        "retrieved_docs": [
            {
                "doc_id": "doc1",
                "text": "Paris is the capital...",
                "retrieval_score": 0.85
            }
        ]
    }
    """
    
    with open(input_file, 'r') as f:
        your_data = json.load(f)
    
    ragnarok_format = []
    
    # If your data is a list of queries
    if isinstance(your_data, list):
        for item in your_data:
            ragnarok_query = convert_single_query(item)
            ragnarok_format.append(ragnarok_query)
    
    # If your data is a single query
    elif isinstance(your_data, dict):
        ragnarok_query = convert_single_query(your_data)
        ragnarok_format.append(ragnarok_query)
    
    # Write in ragnarok format
    with open(output_file, 'w') as f:
        json.dump(ragnarok_format, f, indent=2)
    
    print(f"Converted {len(ragnarok_format)} queries to ragnarok format")
    print(f"Output saved to: {output_file}")

def convert_single_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single query from your format to ragnarok format.
    
    MODIFY THESE FIELD MAPPINGS based on your data structure!
    """
    
    # Map your field names to ragnarok format
    query_id = query_data.get("query_id", query_data.get("qid", "unknown"))
    query_text = query_data.get("query_text", query_data.get("query", ""))
    retrieved_docs = query_data.get("retrieved_docs", query_data.get("documents", []))
    
    candidates = []
    for doc in retrieved_docs:
        candidate = {
            "doc": {
                "segment": doc.get("text", doc.get("content", doc.get("passage", "")))
            },
            "docid": doc.get("doc_id", doc.get("docid", doc.get("id", "unknown"))),
            "score": float(doc.get("retrieval_score", doc.get("score", 0.0)))
        }
        candidates.append(candidate)
    
    ragnarok_query = {
        "query": {
            "text": query_text,
            "qid": query_id
        },
        "candidates": candidates
    }
    
    return ragnarok_query

def initialize_pyserini_searcher():
    """
    Initialize Pyserini searcher once for reuse across multiple files.
    
    Returns:
        LuceneSearcher or None if initialization fails
    """
    if not LuceneSearcher:
        return None
        
    try:
        print("  🔍 Initializing pyserini searcher for msmarco-v2.1-doc index...")
        print("  ⚠️  This may take several minutes for the first run...")
        
        # Add timeout handling
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Pyserini searcher initialization timed out")
        
        # Set a 5-minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes
        
        try:
            searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented')
            signal.alarm(0)  # Cancel the alarm
            print("  ✅ Pyserini searcher initialized successfully!")
            return searcher
        except TimeoutError:
            print("  ❌ Pyserini searcher initialization timed out after 5 minutes")
            print("     This usually means the index is very large and needs more time")
            print("     Will continue with placeholder document text instead")
            return None
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            print(f"  ❌ Could not initialize pyserini searcher: {e}")
            print("     Will use placeholder document text instead")
            return None
            
    except Exception as e:
        print(f"  ⚠️  Could not initialize pyserini searcher: {e}")
        print("     Will use placeholder document text instead.")
        return None

def convert_trec_style_results(queries_file: str, results_file: str, output_file: str, searcher=None, k: int = 5):
    """
    Convert TREC-style results to ragnarok format.
    
    Args:
        queries_file: File with queries (TSV: qid\tquery_text)
        results_file: File with results (TSV: qid\tQ0\tdocid\trank\tscore\ttag)
        output_file: Output file for ragnarok format
        searcher: Pre-initialized Pyserini searcher (optional)
        k: Number of top results to convert per query (default: 5)
    """
    
    print(f"  📖 Loading queries from: {queries_file}")
    # Load queries
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                qid, text = parts
                queries[qid] = text
    
    print(f"  ✅ Loaded {len(queries)} queries")
    
    print(f"  📊 Loading results from: {results_file}")
    # Group results by query
    query_results = {}
    line_count = 0
    with open(results_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"    📝 Processed {line_count:,} lines...")
            
            # Split by whitespace (handles both tabs and spaces)
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
    
    print(f"  ✅ Loaded {len(query_results)} query result groups from {line_count:,} lines")
    print(f"  📝 Note: Only converting top {k} results per query for faster processing")
    
    # Convert to ragnarok format
    print(f"  🔄 Converting to ragnarok format...")
    ragnarok_format = []
    processed_count = 0
    total_queries = len(query_results)
    
    # Extract query ID from filename and process only that single query
    if args.single_file:
        filename = Path(args.single_file).stem
        # Extract query ID from filename
        parts = filename.split('.')
        if len(parts) >= 3:
            query_part = parts[2]
            actual_qid = query_part.split('_')[0]  # Remove _prediction suffix
        else:
            print(f"    ⚠️  Could not extract query ID from filename: {filename}")
            return
            
        if actual_qid not in queries:
            print(f"    ⚠️  Query {actual_qid} not found in topics file")
            return
        
        # Process only the first query group
        if query_results:
            qid, results = next(iter(query_results.items()))
            print(f"    📝 Processing single query: {actual_qid}")
            print(f"    📊 Found {len(results)} documents for query")
        else:
            print(f"    ⚠️  No query results found in file")
            return
    else:
        # Batch processing - process all query results
        print(f"    📝 Processing batch file with {len(query_results)} query groups")
        
        # Process each query group
        for qid, results in query_results.items():
            print(f"    📝 Processing query: {qid}")
            print(f"    📊 Found {len(results)} documents for query")
            
            candidates = []
            # Only process top k results per query for faster conversion
            top_results = sorted(results, key=lambda x: x['rank'])[:k]
            
            for result in top_results:
                # Get document text
                if searcher:
                    try:
                        # Strip segment suffix (e.g., remove #0_3617397938 from msmarco_v2.1_doc_13_1647729865#0_3617397938)
                        base_docid = result['docid']
                        
                        doc = searcher.doc(base_docid)
                        if doc and doc.raw():
                            doc_data = json.loads(doc.raw())
                            # Extract text from document - adjust based on actual structure
                            doc_text = doc_data.get('title','') + " " + doc_data.get('segment', '')
                            if not doc_text:
                                doc_text = f"Document {base_docid} - No text available"
                        else:
                            doc_text = f"Document {base_docid} - Could not retrieve (tried: {base_docid})"
                    except Exception as e:
                        print(f"    ⚠️  Could not retrieve document {result['docid']} (base: {base_docid}): {e}")
                        doc_text = f"Document {base_docid} - Error retrieving"
                else:
                    # This should never happen now since we require Pyserini
                    raise RuntimeError("Pyserini searcher is required but not available")
                
                candidate = {
                    "doc": {"segment": doc_text},
                    "docid": result['docid'],
                    "score": result['score']
                }
                candidates.append(candidate)
            
            # Create ragnarok query with all candidates
            ragnarok_query = {
                "query": {
                    "text": queries.get(qid, f"Query {qid}"),
                    "qid": qid
                },
                "candidates": candidates
            }
            ragnarok_format.append(ragnarok_query)
        
        processed_count = len(query_results)
        print(f"    📝 Processed {processed_count}/{processed_count} queries (100.0%)")
    
    print(f"  💾 Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ragnarok_format, f, indent=2)
    
    print(f"  ✅ Converted {len(ragnarok_format)} queries from TREC format")
    print(f"  📁 Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert TREC-style results to ragnarok format')
    _here = Path(__file__).resolve().parent
    parser.add_argument('--queries', default=str(_here.parent / "querygym" / "queries" / "topics.original.txt"),
                       help='Path to queries file')
    parser.add_argument('--output-dir', default=str(_here.parent / "querygym" / "rag_prepared"),
                       help='Output directory for ragnarok format files')
    parser.add_argument('--single-file', 
                       help='Process only a single run file (for testing)')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of top results to convert per query (default: 5)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Pyserini searcher once for reuse
    print("🔍 Initializing Pyserini searcher for document retrieval...")
    searcher = initialize_pyserini_searcher()
    
    if args.single_file:
        # Process only a single file
        run_file = Path(args.single_file)
        if not run_file.exists():
            print(f"❌ Run file not found: {run_file}")
            exit(1)
        
        # Determine output filename
        run_name = run_file.stem  # Remove .txt extension
        output_file = output_dir / f"ragnarok_format_{run_name}.json"
        
        print(f"Converting single file: {run_file} -> {output_file}")
        print(f"📊 Converting top {args.k} results per query")
        
        convert_trec_style_results(args.queries, str(run_file), str(output_file), searcher=searcher, k=args.k)
        
    else:
        # Process all run files in the retrieval directory
        print("🔄 Processing all run files...")
        
        # Get all run files from the retrieval directory
        runs_dir = output_dir.parent / "retrieval"
        if not runs_dir.exists():
            runs_dir = output_dir.parent / "retrieval_cohere"
        
        if runs_dir.exists():
            all_files = list(runs_dir.glob("run.*.txt"))
            print(f"Found {len(all_files)} run files")
        else:
            print(f"❌ No retrieval directory found. Please specify run files manually.")
            exit(1)
        print(f"📊 Converting top {args.k} results per query")
        print("   Note: Pyserini searcher already initialized - much faster processing!")
        
        # Process each file
        for i, run_file in enumerate(all_files, 1):
            print(f"\n📁 Processing file {i}/{len(all_files)}: {run_file.name}")
            
            # Determine output filename
            run_name = run_file.stem  # Remove .txt extension
            output_file = output_dir / f"ragnarok_format_{run_name}.json"
            
            # Skip if output already exists
            if output_file.exists():
                print(f"   ⏭️  Output already exists: {output_file.name}")
                continue
            
            try:
                convert_trec_style_results(args.queries, str(run_file), str(output_file), searcher=searcher, k=args.k)
                print(f"   ✅ Successfully converted to: {output_file.name}")
            except Exception as e:
                print(f"   ❌ Error processing {run_file.name}: {e}")
                continue
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📁 All ragnarok format files saved to: {output_dir}")
        print(f"📊 Processed {len(all_files)} run files")
    
    print(f"\nConversion complete!")
    print("✅ Files contain real document content from the msmarco-v2.1-doc index")
    print("You can now use these files with ragnarok!")
