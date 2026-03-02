#!/usr/bin/env python3
"""
Run generation-only (no retrieval) on queries from RAG results.
Simple version - just generates answers for queries without citations.
"""

import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

# Add ragnarok to path
sys.path.insert(0, str(_REPO / 'ragnarok/src'))

from ragnarok.data import Query
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.llm import PromptMode


def extract_queries_from_rag_results(rag_results_file):
    """Extract queries from RAG results file."""
    queries = []
    
    # Try JSON first (array format)
    try:
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                queries.append({
                    'qid': item.get('topic_id', ''),
                    'text': item.get('topic', '')
                })
    except json.JSONDecodeError:
        # Try JSONL format
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    queries.append({
                        'qid': item.get('topic_id', ''),
                        'text': item.get('topic', '')
                    })
                except json.JSONDecodeError:
                    continue
    
    return queries


def process_single_file_worker(args_tuple):
    """Worker function to process a single file."""
    rag_file_path, output_dir, model_name, api_key, worker_id, progress_file = args_tuple
    
    start_time = time.time()
    rag_file = Path(rag_file_path)
    
    try:
        # Extract base name
        base_name = rag_file.stem.replace("rag_results_", "")
        output_file = Path(output_dir) / f"rag_results_{base_name}_top0.json"
        
        # Skip if already exists
        if output_file.exists():
            # Log skip
            with open(progress_file, 'a') as f:
                f.write(f"⏭️  [Worker {worker_id}] Skipped (exists): {rag_file.name}\n")
            return True, rag_file.name, None, time.time() - start_time, True
        
        # Log start
        with open(progress_file, 'a') as f:
            f.write(f"🚀 [Worker {worker_id}] Starting: {rag_file.name}\n")
        print(f"[Worker {worker_id}] 🚀 Starting: {rag_file.name}")
        
        # Setup API key for this worker
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Setup model for this worker
        agent = SafeOpenai(
            model=model_name,
            context_size=8192,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=1024,
            num_few_shot_examples=0,
            keys=[api_key],
        )
        
        # Extract queries
        queries = extract_queries_from_rag_results(str(rag_file))
        
        if not queries:
            return False, rag_file.name, "No queries found", time.time() - start_time, False
        
        print(f"[Worker {worker_id}] ✅ Loaded {len(queries)} queries")
        
        # Generate answers for each query
        results = []
        for query in queries:
            try:
                # Create simple prompt - just the question, no retrieval context
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based on your knowledge."},
                    {"role": "user", "content": query['text']}
                ]
                
                # Generate answer using OpenAI API directly
                import openai
                openai.api_key = api_key
                
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7
                )
                
                answer_text = response.choices[0].message.content
                
                # Create result in TREC RAG format (no citations for generation-only)
                result = {
                    "run_id": f"generation_only_{model_name}",
                    "topic_id": query['qid'],
                    "topic": query['text'],
                    "references": [],  # No retrieval, no references
                    "response_length": len(answer_text.split()),
                    "answer": [
                        {"text": answer_text, "citations": []}
                    ]
                }
                results.append(result)
                
            except Exception as e:
                print(f"[Worker {worker_id}] ⚠️  Error processing query {query['qid']}: {e}")
                continue
        
        if not results:
            return False, rag_file.name, "No results generated", time.time() - start_time, False
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        
        # Log completion
        with open(progress_file, 'a') as f:
            f.write(f"✅ [Worker {worker_id}] Completed: {rag_file.name} ({len(results)} queries, {processing_time:.1f}s)\n")
        print(f"[Worker {worker_id}] 🎉 Completed {rag_file.name} in {processing_time:.1f}s")
        
        return True, rag_file.name, None, processing_time, False
        
    except Exception as e:
        return False, rag_file.name, str(e), time.time() - start_time, False


def process_directory_parallel(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "gpt-4o",
    api_key: str = None,
    num_workers: int = 8,
    progress_file: str = None
):
    """Process all RAG result files in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create progress file
    if progress_file is None:
        progress_file = output_dir / "generation_progress.txt"
    
    # Clear progress file
    with open(progress_file, 'w') as f:
        f.write(f"Generation-Only Progress Log\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n")
    
    # Get all JSON files
    json_files = sorted(input_dir.glob("rag_results_*.json"))
    
    if not json_files:
        print(f"⚠️  No RAG result files found in {input_dir}")
        return 0, 0
    
    print(f"📁 Found {len(json_files)} files to process")
    print(f"⚡ Using {num_workers} parallel workers")
    print(f"📝 Progress log: {progress_file}")
    
    # Prepare worker arguments
    worker_args = []
    for i, json_file in enumerate(json_files):
        worker_args.append((
            str(json_file),
            str(output_dir),
            model_name,
            api_key,
            i + 1,
            str(progress_file)
        ))
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file_worker, worker_args),
            total=len(worker_args),
            desc="Processing files",
            unit="file"
        ))
    
    # Count successes and failures
    successful = sum(1 for success, _, _, _, skipped in results if success and not skipped)
    failed = sum(1 for success, _, _, _, _ in results if not success)
    skipped = sum(1 for _, _, _, _, was_skipped in results if was_skipped)
    
    print(f"✅ Successful: {successful}, ❌ Failed: {failed}, ⏭️  Skipped: {skipped}")
    
    return successful, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run generation-only (no retrieval) on RAG result queries'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(_REPO / 'querygym/rag_results/retrieval'),
        help='Directory containing RAG result files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(_REPO / 'querygym/rag_results_o'),
        help='Output directory for generation-only results'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("🚀 Running Generation-Only (No Retrieval)")
    print("=" * 80)
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Workers: {args.num_workers}")
    print()
    
    # Set API key
    os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Process files
    progress_file = output_dir / "generation_progress.txt"
    start_time = time.time()
    success, failed = process_directory_parallel(
        input_dir,
        output_dir,
        args.model,
        args.api_key,
        args.num_workers,
        str(progress_file)
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("🎉 Generation-Only Complete!")
    print("=" * 80)
    print(f"✅ Successful: {success}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"⚡ Average time per file: {total_time/max(success + failed, 1):.1f} seconds")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
