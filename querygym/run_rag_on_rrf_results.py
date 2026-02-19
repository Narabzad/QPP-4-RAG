#!/usr/bin/env python3
"""
Run RAG on RRF retrieval results and evaluate with nuggetizer.

Pipeline:
1. Convert RRF TREC files to ragnarok format (using original queries)
2. Run RAG with GPT-4o on ragnarok format files
3. Evaluate generated answers using nuggetizer
"""

import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime

# Set up Java environment before importing pyserini
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
except ImportError:
    print("❌ ERROR: Pyserini not found!")
    sys.exit(1)

# Add ragnarok to path
sys.path.append('/future/u/negara/home/set_based_QPP/ragnarok/src')
from ragnarok.data import read_requests_from_file
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import PromptMode

def load_original_queries(consolidated_file: str) -> Dict[str, str]:
    """Load original queries from consolidated_query_data.json."""
    print("📖 Loading original queries from consolidated data...")
    with open(consolidated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = {}
    for qid, query_data in data.items():
        queries[qid] = query_data.get('query', '')
    
    print(f"✅ Loaded {len(queries)} original queries")
    return queries

def load_trec_results(results_file: str, k: int = 20) -> Dict[str, List[Dict]]:
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
                       searcher: LuceneSearcher,
                       k: int = 20) -> List[Dict]:
    """Convert to ragnarok format using provided searcher."""
    ragnarok_format = []
    
    for qid, results in query_results.items():
        if qid not in queries:
            continue
        
        candidates = []
        for result in results[:k]:  # Use top k
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
                doc_text = f"Document {base_docid} - Error retrieving: {str(e)[:50]}"
            
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

def convert_rrf_files_to_ragnarok(
    rrf_dir: Path,
    output_dir: Path,
    queries: Dict[str, str],
    searcher: LuceneSearcher,
    k: int = 20
):
    """Convert all RRF TREC files to ragnarok format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rrf_files = sorted(rrf_dir.glob("run.rrf_*.txt"))
    print(f"📁 Found {len(rrf_files)} RRF files to convert")
    
    converted = 0
    skipped = 0
    
    for rrf_file in tqdm(rrf_files, desc="Converting RRF files"):
        output_file = output_dir / f"ragnarok_format_{rrf_file.stem}.json"
        
        # Skip if already exists
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            # Load TREC results
            query_results = load_trec_results(str(rrf_file), k=k)
            
            if not query_results:
                print(f"⚠️  No results in {rrf_file.name}")
                continue
            
            # Convert to ragnarok format
            ragnarok_data = convert_to_ragnarok(queries, query_results, searcher, k=k)
            
            if not ragnarok_data:
                print(f"⚠️  No ragnarok data for {rrf_file.name}")
                continue
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ragnarok_data, f, indent=2)
            
            converted += 1
            
        except Exception as e:
            print(f"❌ Error converting {rrf_file.name}: {e}")
            continue
    
    print(f"✅ Converted {converted} files, skipped {skipped} (already exist)")
    return converted

def run_rag_on_files(
    ragnarok_dir: Path,
    output_dir: Path,
    model_name: str = "gpt-4o",
    api_key: str = None,
    topk: int = 20,
    num_workers: int = 4
):
    """Run RAG on all ragnarok format files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ragnarok_files = sorted(ragnarok_dir.glob("ragnarok_format_*.json"))
    print(f"📁 Found {len(ragnarok_files)} ragnarok files to process")
    
    if not ragnarok_files:
        print("⚠️  No ragnarok files found!")
        return 0, 0
    
    # Set API key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not provided and OPENAI_API_KEY not set!")
        return 0, 0
    
    successful = 0
    failed = 0
    
    for ragnarok_file in tqdm(ragnarok_files, desc="Running RAG"):
        output_file = output_dir / f"rag_results_{ragnarok_file.stem.replace('ragnarok_format_', '')}_{model_name.replace('-', '_')}_top{topk}.json"
        
        # Skip if already exists
        if output_file.exists():
            successful += 1
            continue
        
        try:
            # Load requests
            requests = read_requests_from_file(str(ragnarok_file))
            
            if not requests:
                print(f"⚠️  No requests in {ragnarok_file.name}")
                failed += 1
                continue
            
            # Setup model
            agent = SafeOpenai(
                model=model_name,
                context_size=8192,
                prompt_mode=PromptMode.CHATQA,
                max_output_tokens=1024,
                num_few_shot_examples=0,
                keys=[os.getenv("OPENAI_API_KEY")],
            )
            
            # Extract run name
            base_name = ragnarok_file.stem.replace("ragnarok_format_", "")
            run_name = f"rrf_{base_name}"
            
            # Create RAG system
            rag = RAG(agent=agent, run_id=run_name)
            
            # Process with RAG
            rag_results = rag.answer_batch(
                requests,
                topk=topk,
                shuffle_candidates=False,
                logging=False,
                vllm=False,
            )
            
            if not rag_results:
                print(f"⚠️  No results generated for {ragnarok_file.name}")
                failed += 1
                continue
            
            # Write results
            temp_file = rag.write_answer_results(
                f"temp_{model_name.replace('-', '_')}",
                rag_results,
                shuffle_candidates=False,
                top_k_candidates=topk,
                dataset_name="rrf_dataset",
            )
            
            # Move to final location
            if os.path.exists(temp_file):
                import shutil
                shutil.move(temp_file, str(output_file))
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {ragnarok_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"✅ RAG completed: {successful} successful, {failed} failed")
    return successful, failed

def run_nuggetizer_evaluation(
    rag_results_dir: Path,
    nugget_file: str,
    output_dir: Path,
    log_file: str
):
    """Run nuggetizer evaluation on RAG results using subprocess."""
    print("\n" + "=" * 80)
    print("Running Nuggetizer Evaluation")
    print("=" * 80)
    
    # Use the existing run_rag_nuggetizer.py script via subprocess
    script_path = Path("/future/u/negara/home/set_based_QPP/querygym/run_rag_nuggetizer.py")
    
    if not script_path.exists():
        print(f"❌ Nuggetizer script not found: {script_path}")
        return 0, 1
    
    cmd = [
        "python3",
        str(script_path),
        "--rag-results-dir", str(rag_results_dir),
        "--nugget-file", nugget_file,
        "--output-dir", str(output_dir),
        "--log-file", log_file,
        "--model", "gpt-4o"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Nuggetizer evaluation completed")
        return 1, 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Nuggetizer evaluation failed: {e.stderr[:500]}")
        return 0, 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run RAG on RRF retrieval results and evaluate with nuggetizer'
    )
    parser.add_argument(
        '--rrf-pyserini-dir',
        type=str,
        default='retrieval_RRF_pyserini',
        help='Directory containing pyserini RRF results'
    )
    parser.add_argument(
        '--rrf-cohere-dir',
        type=str,
        default='retrieval_RRF_cohere',
        help='Directory containing cohere RRF results'
    )
    parser.add_argument(
        '--consolidated-file',
        type=str,
        default='consolidated_query_data.json',
        help='Path to consolidated query data JSON file'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for RAG (default: gpt-4o)'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=20,
        help='Number of top documents to use for RAG (default: 20)'
    )
    parser.add_argument(
        '--nugget-file',
        type=str,
        default='/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl',
        help='Path to nugget file for evaluation'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/future/u/negara/home/set_based_QPP/querygym',
        help='Base directory for all operations'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    print("=" * 80)
    print("🚀 Running RAG on RRF Results Pipeline")
    print("=" * 80)
    print(f"📁 Base directory: {base_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Top K documents: {args.topk}")
    print()
    
    # Step 1: Load original queries
    consolidated_file = base_dir / args.consolidated_file
    if not consolidated_file.exists():
        print(f"❌ Consolidated file not found: {consolidated_file}")
        return
    
    queries = load_original_queries(str(consolidated_file))
    
    # Step 2: Initialize pyserini searcher
    print("\n🔍 Initializing Pyserini searcher...")
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2.1-doc-segmented')
        print("✅ Pyserini searcher initialized!\n")
    except Exception as e:
        print(f"❌ Failed to initialize searcher: {e}")
        return
    
    # Step 3: Convert RRF files to ragnarok format
    print("=" * 80)
    print("STEP 1: Converting RRF Files to Ragnarok Format")
    print("=" * 80)
    
    rrf_dirs = [
        (base_dir / args.rrf_pyserini_dir, base_dir / "rag_prepared_RRF" / "retrieval_RRF_pyserini"),
        (base_dir / args.rrf_cohere_dir, base_dir / "rag_prepared_RRF" / "retrieval_RRF_cohere"),
    ]
    
    total_converted = 0
    for rrf_dir, output_dir in rrf_dirs:
        if not rrf_dir.exists():
            print(f"⚠️  RRF directory not found: {rrf_dir}")
            continue
        
        print(f"\n📁 Processing: {rrf_dir.name}")
        converted = convert_rrf_files_to_ragnarok(
            rrf_dir,
            output_dir,
            queries,
            searcher,
            k=args.topk
        )
        total_converted += converted
    
    # Step 4: Run RAG on ragnarok files
    print("\n" + "=" * 80)
    print("STEP 2: Running RAG on Ragnarok Format Files")
    print("=" * 80)
    
    rag_output_dirs = [
        (base_dir / "rag_prepared_RRF" / "retrieval_RRF_pyserini", 
         base_dir / "rag_results_RRF" / "retrieval_RRF_pyserini"),
        (base_dir / "rag_prepared_RRF" / "retrieval_RRF_cohere",
         base_dir / "rag_results_RRF" / "retrieval_RRF_cohere"),
    ]
    
    total_rag_success = 0
    total_rag_failed = 0
    
    for ragnarok_dir, output_dir in rag_output_dirs:
        if not ragnarok_dir.exists():
            print(f"⚠️  Ragnarok directory not found: {ragnarok_dir}")
            continue
        
        print(f"\n📁 Processing: {ragnarok_dir.name}")
        success, failed = run_rag_on_files(
            ragnarok_dir,
            output_dir,
            model_name=args.model,
            api_key=args.api_key,
            topk=args.topk
        )
        total_rag_success += success
        total_rag_failed += failed
    
    # Step 5: Run nuggetizer evaluation
    print("\n" + "=" * 80)
    print("STEP 3: Running Nuggetizer Evaluation")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_dir / f"rag_rrf_nuggetizer_log_{timestamp}.txt"
    
    nugget_output_dirs = [
        (base_dir / "rag_results_RRF" / "retrieval_RRF_pyserini",
         base_dir / "rag_nuggetized_eval_RRF" / "retrieval_RRF_pyserini"),
        (base_dir / "rag_results_RRF" / "retrieval_RRF_cohere",
         base_dir / "rag_nuggetized_eval_RRF" / "retrieval_RRF_cohere"),
    ]
    
    total_nugget_success = 0
    total_nugget_failed = 0
    
    for rag_results_dir, output_dir in nugget_output_dirs:
        if not rag_results_dir.exists():
            print(f"⚠️  RAG results directory not found: {rag_results_dir}")
            continue
        
        print(f"\n📁 Processing: {rag_results_dir.name}")
        success, failed = run_nuggetizer_evaluation(
            rag_results_dir,
            args.nugget_file,
            output_dir,
            str(log_file)
        )
        total_nugget_success += success
        total_nugget_failed += failed
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Pipeline Complete!")
    print("=" * 80)
    print(f"✅ RRF files converted: {total_converted}")
    print(f"✅ RAG successful: {total_rag_success}, failed: {total_rag_failed}")
    print(f"✅ Nuggetizer successful: {total_nugget_success}, failed: {total_nugget_failed}")
    print(f"📝 Log file: {log_file}")
    print(f"📁 Results saved to:")
    print(f"   - Ragnarok format: {base_dir / 'rag_prepared_RRF'}")
    print(f"   - RAG results: {base_dir / 'rag_results_RRF'}")
    print(f"   - Nuggetizer eval: {base_dir / 'rag_nuggetized_eval_RRF'}")

if __name__ == "__main__":
    main()
