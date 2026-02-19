#!/usr/bin/env python3
"""
Run nuggetizer evaluation on RRF RAG results.
This script adapts run_rag_nuggetizer.py to work with RRF directory structure.
"""

import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path("/future/u/negara/home/set_based_QPP/querygym")
    rag_results_dir = base_dir / "rag_results_RRF"
    nugget_file = "/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl"
    output_dir = base_dir / "rag_nuggetized_eval_RRF"
    
    # Import the nuggetizer functions
    sys.path.append(str(base_dir))
    from run_rag_nuggetizer import run_nugget_assignment, run_scoring, log_progress
    from datetime import datetime
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_dir / f"rag_rrf_nuggetizer_log_{timestamp}.txt"
    
    log_progress(str(log_file), "="*80)
    log_progress(str(log_file), "Starting nuggetizer evaluation for RRF RAG results...")
    log_progress(str(log_file), "="*80)
    
    assign_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/assign_nuggets.py")
    score_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/calculate_metrics.py")
    
    # Verify paths
    if not rag_results_dir.exists():
        log_progress(str(log_file), f"❌ Error: RAG results directory not found: {rag_results_dir}")
        return False
    
    pyserini_dir = rag_results_dir / "retrieval_RRF_pyserini"
    cohere_dir = rag_results_dir / "retrieval_RRF_cohere"
    
    if not pyserini_dir.exists() and not cohere_dir.exists():
        log_progress(str(log_file), f"❌ Error: Neither pyserini nor cohere directories found")
        return False
    
    log_progress(str(log_file), f"✅ All paths verified")
    log_progress(str(log_file), f"📁 RAG Results: {rag_results_dir}")
    log_progress(str(log_file), f"📁 Nugget File: {nugget_file}")
    log_progress(str(log_file), f"📁 Output Directory: {output_dir}")
    
    total_assignment_success = 0
    total_assignment_failed = 0
    total_scoring_success = 0
    total_scoring_failed = 0
    
    # Process pyserini folder
    if pyserini_dir.exists():
        assignments_dir = output_dir / "retrieval_RRF_pyserini" / "assignments"
        scores_dir = output_dir / "retrieval_RRF_pyserini" / "scores"
        
        log_progress(str(log_file), f"\n{'='*80}")
        log_progress(str(log_file), "PROCESSING RETRIEVAL_RRF_PYSERINI FOLDER")
        log_progress(str(log_file), f"{'='*80}")
        
        assignment_success, assignment_failed = run_nugget_assignment(
            str(pyserini_dir), nugget_file, str(assignments_dir), 
            str(assign_script), str(score_script), str(scores_dir), 
            str(log_file), model="gpt-4o"
        )
        total_assignment_success += assignment_success
        total_assignment_failed += assignment_failed
        
        # Run catch-up scoring
        if assignment_success > 0:
            log_progress(str(log_file), f"\n{'='*80}")
            log_progress(str(log_file), "CATCH-UP SCORING FOR RETRIEVAL_RRF_PYSERINI")
            log_progress(str(log_file), f"{'='*80}")
            
            scoring_success, scoring_failed = run_scoring(
                str(assignments_dir), str(scores_dir), str(score_script), str(log_file)
            )
            total_scoring_success += scoring_success
            total_scoring_failed += scoring_failed
    
    # Process cohere folder
    if cohere_dir.exists():
        assignments_dir = output_dir / "retrieval_RRF_cohere" / "assignments"
        scores_dir = output_dir / "retrieval_RRF_cohere" / "scores"
        
        log_progress(str(log_file), f"\n{'='*80}")
        log_progress(str(log_file), "PROCESSING RETRIEVAL_RRF_COHERE FOLDER")
        log_progress(str(log_file), f"{'='*80}")
        
        assignment_success, assignment_failed = run_nugget_assignment(
            str(cohere_dir), nugget_file, str(assignments_dir),
            str(assign_script), str(score_script), str(scores_dir),
            str(log_file), model="gpt-4o"
        )
        total_assignment_success += assignment_success
        total_assignment_failed += assignment_failed
        
        # Run catch-up scoring
        if assignment_success > 0:
            log_progress(str(log_file), f"\n{'='*80}")
            log_progress(str(log_file), "CATCH-UP SCORING FOR RETRIEVAL_RRF_COHERE")
            log_progress(str(log_file), f"{'='*80}")
            
            scoring_success, scoring_failed = run_scoring(
                str(assignments_dir), str(scores_dir), str(score_script), str(log_file)
            )
            total_scoring_success += scoring_success
            total_scoring_failed += scoring_failed
    
    # Final Summary
    log_progress(str(log_file), f"\n{'='*80}")
    log_progress(str(log_file), "FINAL SUMMARY")
    log_progress(str(log_file), f"{'='*80}")
    
    log_progress(str(log_file), f"Assignment & Inline Scoring: ✅ {total_assignment_success} successful, ❌ {total_assignment_failed} failed")
    log_progress(str(log_file), f"Catch-up Scoring: ✅ {total_scoring_success} successful, ❌ {total_scoring_failed} failed")
    log_progress(str(log_file), f"📁 Output directory: {output_dir}")
    log_progress(str(log_file), f"📝 Complete log saved to: {log_file}")
    
    if total_assignment_failed == 0 and total_scoring_failed == 0:
        log_progress(str(log_file), f"\n🎉 ALL PROCESSES COMPLETED SUCCESSFULLY!")
        return True
    else:
        log_progress(str(log_file), f"\n⚠️  Some processes failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
