#!/usr/bin/env python3
"""
Run RAG with GPT-4o-mini on all prepared ragnarok format files.
Uses top 3 retrieved documents for each query.
Processes files in parallel for faster execution.
"""
import os
import sys
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import time
import shutil

# Add ragnarok to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ragnarok" / "src"))

from ragnarok.data import read_requests_from_file
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import PromptMode

def setup_openai_api_key():
    """Get OpenAI API key from OPENAI_API_KEY environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found")
    return api_key

def process_single_file_worker(args_tuple):
    """
    Worker function to process a single file with RAG.
    
    Args:
        args_tuple: Tuple containing (file_path, output_dir, model_path, topk, worker_id)
    
    Returns:
        tuple: (success: bool, filename: str, error_message: str or None, processing_time: float)
    """
    file_path, output_dir, model_path, topk, worker_id = args_tuple
    
    start_time = time.time()
    
    try:
        print(f"[Worker {worker_id}] 🚀 Starting: {Path(file_path).name}")
        
        # Load the file
        requests = read_requests_from_file(file_path)
        print(f"[Worker {worker_id}] ✅ Loaded {len(requests)} queries from {Path(file_path).name}")
        
        # Setup the model for this worker
        openai_key = setup_openai_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=8192,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=1024,
            num_few_shot_examples=0,
            keys=[openai_key],
        )
        
        # Extract run info from filename: ragnarok_format_run.{reformulator}_{trial}.json
        base_name = Path(file_path).stem  # e.g., "ragnarok_format_run.genqr_ensemble_trial1"
        run_name = base_name.replace("ragnarok_format_", "")  # e.g., "run.genqr_ensemble_trial1"
        
        # Create RAG system
        rag = RAG(agent=agent, run_id=f"{run_name}_{model_path.replace('-', '_')}")
        
        print(f"[Worker {worker_id}] 🔄 Processing {len(requests)} queries with top {topk} candidates...")
        
        # Process the file with RAG - use only top k candidates
        rag_results = rag.answer_batch(
            requests,
            topk=topk,
            shuffle_candidates=False,
            logging=False,  # Disable verbose logging
            vllm=False,
        )
        
        print(f"[Worker {worker_id}] ✅ Generated answers for {len(rag_results)} queries")
        
        # Write results
        output_file = rag.write_answer_results(
            f"custom_results_{model_path.replace('-', '_')}",
            rag_results,
            shuffle_candidates=False,
            top_k_candidates=topk,
            dataset_name="custom_dataset",
        )
        
        # Move to final output directory with proper naming
        final_filename = f"rag_results_{run_name}_{model_path.replace('-', '_')}_top{topk}.json"
        final_output_path = Path(output_dir) / final_filename
        
        shutil.copy2(output_file, final_output_path)
        
        # Clean up temporary file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        processing_time = time.time() - start_time
        
        print(f"[Worker {worker_id}] 🎉 Completed {Path(file_path).name} in {processing_time:.1f}s")
        print(f"[Worker {worker_id}] 📁 Saved to: {final_filename}")
        
        return True, Path(file_path).name, None, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error processing {Path(file_path).name}: {str(e)}"
        print(f"[Worker {worker_id}] ❌ Failed {Path(file_path).name}: {error_msg}")
        return False, Path(file_path).name, error_msg, processing_time

def process_directory_in_parallel(input_dir: str, output_dir: str, model_path: str = "gpt-4o-mini", topk: int = 3, num_workers: int = 4):
    """
    Process all JSON files in a directory in parallel.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save RAG results
        model_path: Model to use for generation
        topk: Number of top documents to use for RAG
        num_workers: Number of parallel workers to use
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    input_path = Path(input_dir)
    json_files = list(input_path.glob("ragnarok_format_*.json"))
    
    if not json_files:
        print(f"⚠️  No ragnarok format JSON files found in {input_dir}")
        return 0, 0
    
    print(f"📁 Processing directory: {input_dir}")
    print(f"   Found {len(json_files)} files")
    
    # Prepare arguments for parallel processing
    worker_args = []
    for i, json_file in enumerate(json_files):
        worker_args.append((
            str(json_file),
            output_dir,
            model_path,
            topk,
            i + 1
        ))
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file_worker, worker_args),
            total=len(worker_args),
            desc=f"  {Path(input_dir).name}",
            unit="file"
        ))
    
    # Count successes and failures
    successful = sum(1 for success, _, _, _ in results if success)
    failed = sum(1 for success, _, _, _ in results if not success)
    
    return successful, failed

def main():
    """Main function to run RAG on all prepared files."""
    import argparse
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run RAG on all prepared ragnarok format files')
    parser.add_argument('--input-dirs', nargs='+',
                       default=[str(_here / "rag_prepared/retrieval_cohere"),
                                str(_here / "rag_prepared/retrieval")],
                       help='Input directories containing prepared JSON files')
    parser.add_argument('--output-dirs', nargs='+',
                       default=[str(_here / "rag_results/retrieval_cohere"),
                                str(_here / "rag_results/retrieval")],
                       help='Output directories for RAG results')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--topk', type=int, default=3,
                       help='Number of top documents to use for RAG (default: 3)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    INPUT_DIRS = [Path(d) for d in args.input_dirs]
    OUTPUT_DIRS = [Path(d) for d in args.output_dirs]
    MODEL_NAME = args.model
    TOPK = args.topk
    NUM_WORKERS = args.num_workers

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("=" * 80)
    print("🚀 Running RAG on Prepared Ragnarok Format Files")
    print("=" * 80)
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"📊 Using top {TOPK} documents for RAG")
    print(f"⚡ Using {NUM_WORKERS} parallel workers")
    print()
    
    # Process both directories
    total_successful = 0
    total_failed = 0
    start_time = time.time()
    
    for input_dir, output_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
        if not input_dir.exists():
            print(f"⚠️  Directory not found: {input_dir}")
            continue
        
        successful, failed = process_directory_in_parallel(
            str(input_dir),
            str(output_dir),
            MODEL_NAME,
            TOPK,
            NUM_WORKERS
        )
        
        total_successful += successful
        total_failed += failed
        print()
    
    # Summary
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("🎉 All Processing Complete!")
    print("=" * 80)
    print(f"✅ Successfully processed: {total_successful} files")
    print(f"❌ Failed: {total_failed} files")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"⚡ Average time per file: {total_time/max(total_successful + total_failed, 1):.1f} seconds")
    print()
    print("📁 Results saved to:")
    for output_dir in OUTPUT_DIRS:
        if output_dir.exists():
            count = len(list(output_dir.glob("rag_results_*.json")))
            print(f"   - {output_dir}: {count} files")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


