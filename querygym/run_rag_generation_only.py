#!/usr/bin/env python3
"""
Run RAG with topk=0 (generation only, no retrieval context) on existing RAG results.
Then evaluate with nuggetizer.
"""

import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import shutil

# Add ragnarok to path
sys.path.append('/future/u/negara/home/set_based_QPP/ragnarok/src')

from ragnarok.data import read_requests_from_file, Request, Query
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import PromptMode

def setup_openai_api_key(api_key=None):
    """Get OpenAI API key."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
    return api_key

def extract_queries_from_rag_results(rag_results_file):
    """Extract queries from RAG results file to create requests with no candidates."""
    requests = []
    
    # Try JSON first (array format)
    try:
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                query = Query(
                    text=item.get('topic', ''),
                    qid=item.get('topic_id', '')
                )
                request = Request(query=query, candidates=[])
                requests.append(request)
        else:
            # Single object
            query = Query(
                text=data.get('topic', ''),
                qid=data.get('topic_id', '')
            )
            request = Request(query=query, candidates=[])
            requests.append(request)
    except json.JSONDecodeError:
        # Try JSONL format (one JSON object per line)
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    query = Query(
                        text=item.get('topic', ''),
                        qid=item.get('topic_id', '')
                    )
                    request = Request(query=query, candidates=[])
                    requests.append(request)
                except json.JSONDecodeError:
                    continue
    
    return requests

def process_single_file_worker(args_tuple):
    """Worker function to process a single file with generation-only RAG."""
    rag_file_path, output_dir, model_name, api_key, worker_id = args_tuple
    
    start_time = time.time()
    rag_file = Path(rag_file_path)
    
    try:
        # Extract base name
        base_name = rag_file.stem.replace("rag_results_", "")
        output_file = Path(output_dir) / f"rag_results_{base_name}_top0.json"
        
        # Skip if already exists
        if output_file.exists():
            return True, rag_file.name, None, time.time() - start_time, True
        
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
        
        # Extract queries from RAG results
        requests = extract_queries_from_rag_results(str(rag_file))
        
        if not requests:
            return False, rag_file.name, "No queries found", time.time() - start_time, False
        
        # Create RAG system
        run_name = base_name.replace("ragnarok_format_", "")
        rag = RAG(agent=agent, run_id=f"{run_name}_{model_name.replace('-', '_')}_top0")
        
        # For generation only, use RagnarokTemplates directly with empty context
        from ragnarok.data import Result, CitedSentence
        from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates
        
        rag_results = []
        for request in requests:
            try:
                # Create prompt using RagnarokTemplates with empty context (no retrieval)
                ragnarok_template = RagnarokTemplates(agent._prompt_mode)
                messages = ragnarok_template(request.query.text, [], "gpt")  # Empty context list
                
                # Use agent's run_llm to generate answer
                answer_text, rag_exec_info = agent.run_llm(messages, logging=False)
                
                # Parse answer_text into CitedSentence format
                answer_sentences = [CitedSentence(text=answer_text, citations=[])]
                
                # Create result with empty references (no retrieval)
                result = Result(
                    query=request.query,
                    references=[],  # No retrieval, so no references
                    answer=answer_sentences,
                    rag_exec_summary=rag_exec_info,
                )
                rag_results.append(result)
            except Exception as e:
                print(f"[Worker {worker_id}] ⚠️  Error processing query {request.query.qid}: {e}")
                continue
        
        if not rag_results:
            return False, rag_file.name, "No results generated", time.time() - start_time, False
        
        # Write results
        temp_file = rag.write_answer_results(
            f"temp_{model_name.replace('-', '_')}",
            rag_results,
            shuffle_candidates=False,
            top_k_candidates=0,
            dataset_name="generation_only",
        )
        
        # Move to final location
        if os.path.exists(temp_file):
            shutil.move(temp_file, str(output_file))
        
        return True, rag_file.name, None, time.time() - start_time, False
        
    except Exception as e:
        return False, rag_file.name, str(e), time.time() - start_time, False

def process_rag_results_for_generation_only(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "gpt-4o",
    api_key: str = None,
    num_workers: int = 4
):
    """Process RAG result files to run generation only (topk=0) in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = sorted(input_dir.glob("rag_results_*.json"))
    
    if not json_files:
        print(f"⚠️  No RAG result files found in {input_dir}")
        return 0, 0
    
    print(f"📁 Processing {len(json_files)} files from {input_dir.name}")
    print(f"⚡ Using {num_workers} parallel workers")
    
    # Setup API key
    setup_openai_api_key(api_key)
    
    # Prepare worker arguments
    worker_args = []
    for i, json_file in enumerate(json_files):
        worker_args.append((
            str(json_file),
            str(output_dir),
            model_name,
            api_key or os.getenv("OPENAI_API_KEY"),
            i + 1
        ))
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file_worker, worker_args),
            total=len(worker_args),
            desc=f"Processing {input_dir.name}",
            unit="file"
        ))
    
    # Count successes and failures
    successful = sum(1 for success, _, _, _, _ in results if success)
    failed = sum(1 for success, _, _, _, _ in results if not success)
    skipped = sum(1 for _, _, _, _, was_skipped in results if was_skipped)
    
    print(f"✅ Successful: {successful}, ❌ Failed: {failed}, ⏭️  Skipped: {skipped}")
    
    return successful, failed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run RAG with topk=0 (generation only) on existing RAG results'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/future/u/negara/home/set_based_QPP/querygym/rag_results',
        help='Base directory containing retrieval and retrieval_cohere folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/future/u/negara/home/set_based_QPP/querygym/rag_results_o',
        help='Output directory for generation-only results'
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
        help='Model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--run-nuggetizer',
        action='store_true',
        help='Run nuggetizer evaluation after RAG'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    
    args = parser.parse_args()
    
    base_input_dir = Path(args.input_dir)
    base_output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("🚀 Running RAG with topk=0 (Generation Only)")
    print("=" * 80)
    print(f"📁 Input: {base_input_dir}")
    print(f"📁 Output: {base_output_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Parallel workers: {args.num_workers}")
    print()
    
    total_success = 0
    total_failed = 0
    
    # Process retrieval folder
    retrieval_input = base_input_dir / "retrieval"
    retrieval_output = base_output_dir / "retrieval"
    
    if retrieval_input.exists():
        print(f"\n📁 Processing: retrieval")
        success, failed = process_rag_results_for_generation_only(
            retrieval_input,
            retrieval_output,
            args.model,
            args.api_key,
            args.num_workers
        )
        total_success += success
        total_failed += failed
    
    # Process retrieval_cohere folder
    cohere_input = base_input_dir / "retrieval_cohere"
    cohere_output = base_output_dir / "retrieval_cohere"
    
    if cohere_input.exists():
        print(f"\n📁 Processing: retrieval_cohere")
        success, failed = process_rag_results_for_generation_only(
            cohere_input,
            cohere_output,
            args.model,
            args.api_key,
            args.num_workers
        )
        total_success += success
        total_failed += failed
    
    print("\n" + "=" * 80)
    print("🎉 RAG Generation-Only Complete!")
    print("=" * 80)
    print(f"✅ Successful: {total_success}")
    print(f"❌ Failed: {total_failed}")
    print(f"📁 Results saved to: {base_output_dir}")
    
    # Run nuggetizer if requested
    if args.run_nuggetizer:
        print("\n" + "=" * 80)
        print("📊 Running Nuggetizer Evaluation")
        print("=" * 80)
        
        from run_rag_nuggetizer import run_nugget_assignment, run_scoring, log_progress
        from datetime import datetime
        
        nugget_file = "/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl"
        output_eval_dir = Path("/future/u/negara/home/set_based_QPP/querygym/rag_nuggetized_eval_o")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("/future/u/negara/home/set_based_QPP/querygym") / f"rag_generation_only_nuggetizer_log_{timestamp}.txt"
        
        assign_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/assign_nuggets.py")
        score_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/calculate_metrics.py")
        
        # Process retrieval folder
        if retrieval_output.exists():
            assignments_dir = output_eval_dir / "retrieval" / "assignments"
            scores_dir = output_eval_dir / "retrieval" / "scores"
            
            log_progress(str(log_file), f"\n{'='*80}")
            log_progress(str(log_file), "PROCESSING RETRIEVAL (GENERATION ONLY)")
            log_progress(str(log_file), f"{'='*80}")
            
            assignment_success, assignment_failed = run_nugget_assignment(
                str(retrieval_output), nugget_file, str(assignments_dir),
                str(assign_script), str(score_script), str(scores_dir),
                str(log_file), model="gpt-4o"
            )
            
            if assignment_success > 0:
                run_scoring(str(assignments_dir), str(scores_dir), str(score_script), str(log_file))
        
        # Process retrieval_cohere folder
        if cohere_output.exists():
            assignments_dir = output_eval_dir / "retrieval_cohere" / "assignments"
            scores_dir = output_eval_dir / "retrieval_cohere" / "scores"
            
            log_progress(str(log_file), f"\n{'='*80}")
            log_progress(str(log_file), "PROCESSING RETRIEVAL_COHERE (GENERATION ONLY)")
            log_progress(str(log_file), f"{'='*80}")
            
            assignment_success, assignment_failed = run_nugget_assignment(
                str(cohere_output), nugget_file, str(assignments_dir),
                str(assign_script), str(score_script), str(scores_dir),
                str(log_file), model="gpt-4o"
            )
            
            if assignment_success > 0:
                run_scoring(str(assignments_dir), str(scores_dir), str(score_script), str(log_file))
        
        print(f"📝 Nuggetizer log: {log_file}")

if __name__ == "__main__":
    main()
