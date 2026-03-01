#!/usr/bin/env python3
"""
Wrapper script to run nuggetizer assignment and scoring for ragnarok results.
This script adapts the run_complete_pipeline.py to work with our pipeline structure.
"""

import os
import subprocess
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def log_progress(log_file, message):
    """Write progress message to log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    with open(log_file, 'a') as f:
        f.write(log_message)
    print(message)  # Also print to console

def run_nugget_assignment(rag_results_dir, nugget_file, output_dir, assign_script, log_file):
    """Run nugget assignment for a specific nugget file."""
    rag_results_dir = Path(rag_results_dir)
    nugget_file = Path(nugget_file)
    output_dir = Path(output_dir)
    assign_script = Path(assign_script)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get expected number of lines from nugget file
    with open(nugget_file, 'r') as f:
        expected_lines = sum(1 for _ in f)
    
    rag_result_files = list(rag_results_dir.glob("*.json"))
    
    # Check if RAG files contain single queries (1 line each) vs full dataset
    # If all RAG files have 1 line, this is a single-query-per-file setup
    single_query_mode = True
    for rag_file in rag_result_files[:3]:  # Check first 3 files
        try:
            with open(rag_file, 'r') as f:
                rag_lines = sum(1 for _ in f)
            if rag_lines > 1:
                single_query_mode = False
                break
        except Exception as e:
            log_progress(log_file, f"Could not read RAG file {rag_file}: {e}")
    
    if single_query_mode:
        expected_lines = 1
        log_progress(log_file, f"Detected single-query-per-file mode, adjusting expected lines to {expected_lines}")
    else:
        log_progress(log_file, f"Detected full-dataset mode, using expected lines from nugget file: {expected_lines}")
    log_progress(log_file, f"Found {len(rag_result_files)} RAG result files to process for nugget assignment.")
    log_progress(log_file, f"Using nugget file: {nugget_file.name} ({expected_lines} queries with nuggets)")
    log_progress(log_file, f"Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, rag_file in enumerate(tqdm(rag_result_files, desc="Assigning Nuggets", unit="file"), 1):
        output_assignment_file = output_dir / f"{rag_file.stem}_assignments.jsonl"
        
        # Skip if file already exists and has correct number of lines
        if output_assignment_file.exists():
            try:
                with open(output_assignment_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                if line_count == expected_lines:
                    log_progress(log_file, f"Skipping {rag_file.name} (assignments already exist with {expected_lines} lines)")
                    successful += 1
                    continue
                else:
                    log_progress(log_file, f"Re-processing {rag_file.name} (existing file has {line_count} lines, expected {expected_lines})")
                    output_assignment_file.unlink()
            except Exception as e:
                log_progress(log_file, f"Error checking {output_assignment_file}: {e}")
                if output_assignment_file.exists():
                    output_assignment_file.unlink()
        
        cmd = [
            "python3", str(assign_script),
            "--nugget_file", str(nugget_file),
            "--answer_file", str(rag_file),
            "--output_file", str(output_assignment_file)
        ]
        
        try:
            log_progress(log_file, f"Processing {rag_file.name} ({i}/{len(rag_result_files)})...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            # Verify the output file has expected number of lines
            if output_assignment_file.exists():
                with open(output_assignment_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                if line_count == expected_lines:
                    log_progress(log_file, f"✅ Successfully assigned nuggets for {rag_file.name} ({line_count} lines)")
                    successful += 1
                else:
                    log_progress(log_file, f"❌ Error: {rag_file.name} has {line_count} lines, expected {expected_lines}")
                    failed += 1
            else:
                log_progress(log_file, f"❌ Error: Output file not created for {rag_file.name}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            log_progress(log_file, f"⏰ Timeout processing {rag_file.name}")
            failed += 1
        except subprocess.CalledProcessError as e:
            log_progress(log_file, f"❌ Error assigning nuggets for {rag_file.name}: {e.stderr}")
            failed += 1
        except Exception as e:
            log_progress(log_file, f"❌ Unexpected error for {rag_file.name}: {e}")
            failed += 1
    
    log_progress(log_file, f"\n🎉 Nugget assignment complete!")
    log_progress(log_file, f"   ✅ Successful: {successful}")
    log_progress(log_file, f"   ❌ Failed: {failed}")
    log_progress(log_file, f"   📁 Assignment files saved to: {output_dir}")
    
    return successful, failed

def run_scoring(assignments_dir, scores_dir, script_path, log_file):
    """Run scoring for assignment files."""
    assignments_dir = Path(assignments_dir)
    scores_dir = Path(scores_dir)
    script_path = Path(script_path)
    
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    assignment_files = list(assignments_dir.glob("*.jsonl"))
    log_progress(log_file, f"Found {len(assignment_files)} assignment files to score")
    
    successful = 0
    failed = 0
    
    for i, assignment_file in enumerate(tqdm(assignment_files, desc="Scoring Files", unit="file"), 1):
        score_file = scores_dir / f"{assignment_file.stem}.jsonl"
        
        # Skip if already processed
        if score_file.exists():
            log_progress(log_file, f"⏭️  [{i}/{len(assignment_files)}] Skipping {assignment_file.name} (scores already exist)")
            successful += 1
            continue
        
        cmd = [
            "python3", str(script_path),
            "--input_file", str(assignment_file),
            "--output_file", str(score_file)
        ]
        
        try:
            log_progress(log_file, f"🔄 [{i}/{len(assignment_files)}] Scoring {assignment_file.name}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                log_progress(log_file, f"✅ Successfully scored {assignment_file.name}")
                successful += 1
            else:
                log_progress(log_file, f"❌ Failed to score {assignment_file.name}")
                log_progress(log_file, f"   Error: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            log_progress(log_file, f"⏰ Timeout scoring {assignment_file.name}")
            failed += 1
        except Exception as e:
            log_progress(log_file, f"💥 Exception scoring {assignment_file.name}: {e}")
            failed += 1
    
    log_progress(log_file, f"\n🎉 Scoring complete!")
    log_progress(log_file, f"   ✅ Successful: {successful}")
    log_progress(log_file, f"   ❌ Failed: {failed}")
    log_progress(log_file, f"   📁 Score files saved to: {scores_dir}")
    
    return successful, failed

def main():
    """Main function to run nuggetizer assignment and scoring for ragnarok results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run nuggetizer assignment and scoring for ragnarok results')
    parser.add_argument('--ragnarok-dir', type=str, required=True,
                       help='Directory containing ragnarok JSON files')
    parser.add_argument('--nugget-file', type=str, required=True,
                       help='Path to nugget file')
    parser.add_argument('--assignments-dir', type=str, required=True,
                       help='Output directory for assignment files')
    parser.add_argument('--scores-dir', type=str, required=True,
                       help='Output directory for score files')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Create log file with timestamp if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(Path.cwd() / f"nuggetizer_log_{timestamp}.txt")
    else:
        log_file = args.log_file

    log_progress(log_file, "Starting nuggetizer assignment and scoring pipeline...")

    # Define paths
    _scripts_dir = Path(__file__).resolve().parent
    _repo_dir = _scripts_dir.parent
    ragnarok_dir = Path(args.ragnarok_dir)
    nugget_file = Path(args.nugget_file)
    assignments_dir = Path(args.assignments_dir)
    scores_dir = Path(args.scores_dir)
    assign_script = _scripts_dir / "assign_nuggets_incremental.py"
    score_script = _repo_dir / "nuggetizer" / "scripts" / "calculate_metrics.py"
    
    # Verify input files exist
    if not ragnarok_dir.exists():
        log_progress(log_file, f"❌ Error: Ragnarok directory not found: {ragnarok_dir}")
        return False
    
    if not nugget_file.exists():
        log_progress(log_file, f"❌ Error: Nugget file not found: {nugget_file}")
        return False
    
    if not assign_script.exists():
        log_progress(log_file, f"❌ Error: Assignment script not found: {assign_script}")
        return False
        
    if not score_script.exists():
        log_progress(log_file, f"❌ Error: Scoring script not found: {score_script}")
        return False
    
    log_progress(log_file, f"✅ All input files verified")
    log_progress(log_file, f"📁 Ragnarok Results: {ragnarok_dir}")
    log_progress(log_file, f"📁 Nugget File: {nugget_file}")
    
    # Run assignment
    log_progress(log_file, f"\n{'='*80}")
    log_progress(log_file, "RUNNING NUGGET ASSIGNMENT")
    log_progress(log_file, f"{'='*80}")
    
    assignment_success, assignment_failed = run_nugget_assignment(
        ragnarok_dir, nugget_file, assignments_dir, assign_script, log_file
    )
    
    # Run scoring if assignment was successful
    if assignment_success > 0:
        log_progress(log_file, f"\n{'='*80}")
        log_progress(log_file, "RUNNING NUGGET SCORING")
        log_progress(log_file, f"{'='*80}")
        
        scoring_success, scoring_failed = run_scoring(
            assignments_dir, scores_dir, score_script, log_file
        )
    else:
        log_progress(log_file, "❌ Skipping scoring due to assignment failures")
        scoring_success, scoring_failed = 0, 0
    
    # Final Summary
    log_progress(log_file, f"\n{'='*80}")
    log_progress(log_file, "FINAL SUMMARY")
    log_progress(log_file, f"{'='*80}")
    
    log_progress(log_file, f"Assignment: ✅ {assignment_success} successful, ❌ {assignment_failed} failed")
    log_progress(log_file, f"Scoring: ✅ {scoring_success} successful, ❌ {scoring_failed} failed")
    log_progress(log_file, f"📁 Assignments: {assignments_dir}")
    log_progress(log_file, f"📁 Scores: {scores_dir}")
    
    if assignment_failed == 0 and scoring_failed == 0:
        log_progress(log_file, f"\n🎉 ALL PROCESSES COMPLETED SUCCESSFULLY!")
        return True
    else:
        log_progress(log_file, f"\n⚠️  Some processes failed. Check the logs above for details.")
        return False
    
    log_progress(log_file, f"\n📝 Complete log saved to: {log_file}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
