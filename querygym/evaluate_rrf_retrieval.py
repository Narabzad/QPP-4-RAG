#!/usr/bin/env python3
"""
Evaluate RRF (Reciprocal Rank Fusion) retrieval results per query.
Computes NDCG@10 and Recall@100 for each query in each RRF run file.
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm

try:
    import pytrec_eval
except ImportError:
    print("❌ pytrec_eval not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytrec_eval"])
    import pytrec_eval


def load_qrels(qrels_file):
    """Load qrels file."""
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    query_id, iter_num, doc_id, relevance = parts[0], parts[1], parts[2], parts[3]
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = int(relevance)
    return qrels


def load_run_file(run_file):
    """Load run file in TREC format."""
    run = {}
    with open(run_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, run_name = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    if query_id not in run:
                        run[query_id] = {}
                    run[query_id][doc_id] = float(score)
    return run


def evaluate_per_query(run_file, qrels, metrics=None):
    """Evaluate a single run file and return per-query metrics."""
    if metrics is None:
        metrics = ['ndcg_cut_10', 'recall_100']
    
    try:
        run = load_run_file(run_file)
        
        # Create evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
        
        # Evaluate
        results = evaluator.evaluate(run)
        
        # Convert to list of records
        per_query_results = []
        for qid in sorted(results.keys()):
            record = {'qid': qid}
            for metric in metrics:
                if metric in results[qid]:
                    record[metric] = results[qid][metric]
                else:
                    record[metric] = 0.0
            per_query_results.append(record)
        
        return per_query_results
    except Exception as e:
        print(f"   ⚠️  Error evaluating {run_file.name}: {e}")
        return None


def get_run_name_from_file(run_file):
    """Extract run name from filename (e.g., run.rrf_IDF_avg.txt -> rrf_IDF_avg)."""
    name = run_file.stem  # Remove .txt
    if name.startswith('run.'):
        return name[4:]  # Remove 'run.' prefix
    return name


def main():
    # Paths
    qrels_file = Path('/future/u/negara/home/set_based_QPP/data/qrels.rag24.raggy-dev.txt')
    rrf_pyserini_dir = Path('/future/u/negara/home/set_based_QPP/querygym/retrieval_RRF_pyserini')
    rrf_cohere_dir = Path('/future/u/negara/home/set_based_QPP/querygym/retrieval_RRF_cohere')
    output_dir = Path('/future/u/negara/home/set_based_QPP/querygym/retrieval_eval_RRF')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load qrels file
    print("📖 Loading qrels file...")
    if not qrels_file.exists():
        print(f"❌ Qrels file not found: {qrels_file}")
        return
    
    qrels = load_qrels(qrels_file)
    print(f"✅ Loaded qrels: {len(qrels)} queries")
    
    # Find all RRF run files
    if not rrf_pyserini_dir.exists():
        print(f"❌ RRF pyserini directory not found: {rrf_pyserini_dir}")
        return
    
    if not rrf_cohere_dir.exists():
        print(f"❌ RRF cohere directory not found: {rrf_cohere_dir}")
        return
    
    pyserini_files = sorted(rrf_pyserini_dir.glob("run.*.txt"))
    cohere_files = sorted(rrf_cohere_dir.glob("run.*.txt"))
    
    print(f"📁 Found {len(pyserini_files)} RRF pyserini run files")
    print(f"📁 Found {len(cohere_files)} RRF cohere run files")
    
    print(f"\n📊 Evaluating RRF run files per query...")
    print("=" * 80)
    
    # Evaluate pyserini RRF files
    successful = 0
    failed = 0
    
    print("\n🔍 Evaluating RRF Pyserini files...")
    for run_file in tqdm(pyserini_files, desc="Evaluating pyserini RRF", unit="file"):
        run_name = get_run_name_from_file(run_file)
        print(f"\n📄 Evaluating: {run_name} (pyserini)")
        
        results = evaluate_per_query(run_file, qrels)
        if results:
            # Save to JSONL file
            output_file = output_dir / f"retrieval_RRF_pyserini_{run_name}_per_query.jsonl"
            with open(output_file, 'w') as f:
                for record in results:
                    f.write(json.dumps(record) + '\n')
            
            # Calculate averages for display
            avg_ndcg = sum(r['ndcg_cut_10'] for r in results) / len(results) if results else 0.0
            avg_recall = sum(r['recall_100'] for r in results) / len(results) if results else 0.0
            print(f"   ✅ Saved {len(results)} queries | Avg NDCG@10: {avg_ndcg:.6f}, Avg Recall@100: {avg_recall:.6f}")
            print(f"   📄 Saved to: {output_file.name}")
            successful += 1
        else:
            print(f"   ❌ Evaluation failed")
            failed += 1
    
    # Evaluate cohere RRF files
    print("\n🔍 Evaluating RRF Cohere files...")
    for run_file in tqdm(cohere_files, desc="Evaluating cohere RRF", unit="file"):
        run_name = get_run_name_from_file(run_file)
        print(f"\n📄 Evaluating: {run_name} (cohere)")
        
        results = evaluate_per_query(run_file, qrels)
        if results:
            # Save to JSONL file
            output_file = output_dir / f"retrieval_RRF_cohere_{run_name}_per_query.jsonl"
            with open(output_file, 'w') as f:
                for record in results:
                    f.write(json.dumps(record) + '\n')
            
            # Calculate averages for display
            avg_ndcg = sum(r['ndcg_cut_10'] for r in results) / len(results) if results else 0.0
            avg_recall = sum(r['recall_100'] for r in results) / len(results) if results else 0.0
            print(f"   ✅ Saved {len(results)} queries | Avg NDCG@10: {avg_ndcg:.6f}, Avg Recall@100: {avg_recall:.6f}")
            print(f"   📄 Saved to: {output_file.name}")
            successful += 1
        else:
            print(f"   ❌ Evaluation failed")
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {output_dir}")
    print("\n✅ Evaluation complete!")
    print(f"📄 Each file contains per-query metrics with 'qid', 'ndcg_cut_10', and 'recall_100' fields")


if __name__ == "__main__":
    main()
