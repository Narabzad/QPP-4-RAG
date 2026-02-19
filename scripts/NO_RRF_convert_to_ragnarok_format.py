#!/usr/bin/env python3
"""
Convert individual retrieval results (not RRF) to ragnarok format.
This script handles single retrieval result files with multiple queries.
"""
import json
import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# Set up Java environment before importing pyserini
def setup_java_environment():
    """Set up Java environment variables needed for pyserini."""
    # Set JAVA_HOME to miniconda3 Java installation
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    # Try to find the correct JVM path
    possible_jvm_paths = [
        "/future/u/negara/miniconda3/lib/server/libjvm.so",
        "/future/u/negara/miniconda3/pkgs/openjdk-21.0.6-h38aa4c6_0/lib/server/libjvm.so",
        "/future/u/negara/miniconda3/envs/pyserini-env/lib/server/libjvm.so"
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
        print("⚠️  Warning: Could not find libjvm.so. Pyserini may not work.")
    
    print(f"✅ Set JAVA_HOME to: {java_home}")

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

def initialize_pyserini_searcher():
    """
    Initialize Pyserini searcher once for reuse across multiple files.
    
    Returns:
        LuceneSearcher or None if initialization fails
    """
    if not LuceneSearcher:
        return None
        
    try:
        print("  🔍 Initializing pyserini searcher for msmarco-v2.1-doc-segmented index...")
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

def convert_trec_style_results(queries_file: str, results_file: str, output_file: str, use_pyserini: bool = True, k: int = 5):
    """
    Convert TREC-style results to ragnarok format.
    
    Args:
        queries_file: File with queries (TSV: qid\tquery_text)
        results_file: File with results (TSV: qid\tQ0\tdocid\trank\tscore\ttag)
        output_file: Output file for ragnarok format
        use_pyserini: Whether to use pyserini to get actual document text
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
    
    # Initialize pyserini searcher if available
    searcher = None
    if use_pyserini and LuceneSearcher:
        try:
            print("  🔍 Initializing pyserini searcher for msmarco-v2.1-doc-segmented index...")
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
            except TimeoutError:
                print("  ❌ Pyserini searcher initialization timed out after 5 minutes")
                print("     This usually means the index is very large and needs more time")
                print("     Will continue with placeholder document text instead")
                searcher = None
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                print(f"  ❌ Could not initialize pyserini searcher: {e}")
                print("     Will use placeholder document text instead")
                searcher = None
                
        except Exception as e:
            print(f"  ⚠️  Could not initialize pyserini searcher: {e}")
            print("     Will use placeholder document text instead.")
            searcher = None
    
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
    
    for qid, results in query_results.items():
        if qid not in queries:
            continue
            
        candidates = []
        # Only process top k results per query for faster conversion
        top_results = sorted(results, key=lambda x: x['rank'])[:k]
        
        for result in top_results:
            # Get document text
            if searcher:
                try:
                    # Strip segment suffix (e.g., remove #0_3617397938 from msmarco_v2.1_doc_13_1647729865#0_3617397938)
                    #base_docid = result['docid'].split('#')[0] if '#' in result['docid'] else result['docid']
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
        
        ragnarok_query = {
            "query": {
                "text": queries[qid],
                "qid": qid
            },
            "candidates": candidates
        }
        ragnarok_format.append(ragnarok_query)
        
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_queries:
            print(f"    📝 Processed {processed_count}/{total_queries} queries ({processed_count/total_queries*100:.1f}%)")
    
    print(f"  💾 Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ragnarok_format, f, indent=2)
    
    print(f"  ✅ Converted {len(ragnarok_format)} queries from TREC format")
    print(f"  📁 Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert individual retrieval results to ragnarok format')
    parser.add_argument('--queries', required=True,
                       help='Path to queries file (TSV: qid\tquery_text)')
    parser.add_argument('--results-file', required=True,
                       help='Path to retrieval results file (TSV: qid\tQ0\tdocid\trank\tscore\ttag)')
    parser.add_argument('--output-file', required=True,
                       help='Output file for ragnarok format JSON')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of top results to convert per query (default: 5)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not Path(args.queries).exists():
        print(f"❌ Queries file not found: {args.queries}")
        sys.exit(1)
    
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting retrieval results: {args.results_file} -> {args.output_file}")
    print("🔍 Using Pyserini to get real document content from msmarco-v2.1-doc-segmented index")
    print(f"📊 Converting top {args.k} results per query")
    print("   Note: This may take several minutes for the first run")
    
    convert_trec_style_results(args.queries, args.results_file, args.output_file, use_pyserini=True, k=args.k)
    
    print(f"\nConversion complete!")
    print("✅ File contains real document content from the msmarco-v2.1-doc-segmented index")
    print("You can now use this file with ragnarok!")
