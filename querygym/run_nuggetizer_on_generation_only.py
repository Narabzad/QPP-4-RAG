#!/usr/bin/env python3
"""
Run nuggetizer evaluation on generation-only results.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Add to path
sys.path.insert(0, str(_REPO / 'querygym'))
from run_rag_nuggetizer import run_nugget_assignment, run_scoring, log_progress
from datetime import datetime

def main():
    base_dir = _REPO / "querygym"
    rag_results_dir = base_dir / "rag_results_o"  # Files are .json but JSONL format inside
    nugget_file = str(_REPO / "data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl")
    output_dir = base_dir / "rag_nuggetized_eval_o"

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_dir / f"rag_generation_only_nuggetizer_log_{timestamp}.txt"

    log_progress(str(log_file), "="*80)
    log_progress(str(log_file), "Starting nuggetizer evaluation for generation-only results...")
    log_progress(str(log_file), "="*80)

    assign_script = _REPO / "nuggetizer/scripts/assign_nuggets.py"
    score_script = _REPO / "nuggetizer/scripts/calculate_metrics.py"
    
    # Verify paths
    if not rag_results_dir.exists():
        log_progress(str(log_file), f"❌ Error: RAG results directory not found: {rag_results_dir}")
        return False
    
    log_progress(str(log_file), f"✅ All paths verified")
    log_progress(str(log_file), f"📁 RAG Results: {rag_results_dir}")
    log_progress(str(log_file), f"📁 Nugget File: {nugget_file}")
    log_progress(str(log_file), f"📁 Output Directory: {output_dir}")
    
    # Process the flat directory structure
    assignments_dir = output_dir / "assignments"
    scores_dir = output_dir / "scores"
    
    log_progress(str(log_file), f"\n{'='*80}")
    log_progress(str(log_file), "PROCESSING GENERATION-ONLY RESULTS")
    log_progress(str(log_file), f"{'='*80}")
    
    assignment_success, assignment_failed = run_nugget_assignment(
        str(rag_results_dir), nugget_file, str(assignments_dir), 
        str(assign_script), str(score_script), str(scores_dir), 
        str(log_file), model="gpt-4o"
    )
    
    # Run catch-up scoring
    if assignment_success > 0:
        log_progress(str(log_file), f"\n{'='*80}")
        log_progress(str(log_file), "CATCH-UP SCORING")
        log_progress(str(log_file), f"{'='*80}")
        
        scoring_success, scoring_failed = run_scoring(
            str(assignments_dir), str(scores_dir), str(score_script), str(log_file)
        )
    
    # Final Summary
    log_progress(str(log_file), f"\n{'='*80}")
    log_progress(str(log_file), "FINAL SUMMARY")
    log_progress(str(log_file), f"{'='*80}")
    
    log_progress(str(log_file), f"Assignment & Inline Scoring: ✅ {assignment_success} successful, ❌ {assignment_failed} failed")
    log_progress(str(log_file), f"📁 Output directory: {output_dir}")
    log_progress(str(log_file), f"📝 Complete log saved to: {log_file}")
    
    if assignment_failed == 0:
        log_progress(str(log_file), f"\n🎉 ALL PROCESSES COMPLETED SUCCESSFULLY!")
        return True
    else:
        log_progress(str(log_file), f"\n⚠️  Some processes failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
