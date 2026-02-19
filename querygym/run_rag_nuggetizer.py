#!/usr/bin/env python3
"""
Script to run nuggetizer assignment and scoring for RAG results.
Processes all JSON files in retrieval and retrieval_cohere folders.
"""

import os
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from nuggetizer.core.metrics import calculate_nugget_scores

def log_progress(log_file, message):
    """Write progress message to log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    with open(log_file, 'a') as f:
        f.write(log_message)
    print(message)  # Also print to console

def calculate_and_append_metrics(assignment_file, score_file, expected_lines):
    """Calculate metrics for all records in assignment file and append to score file incrementally."""
    # Read all records from assignment file
    assignment_records = []
    try:
        with open(assignment_file, 'r') as f:
            for line in f:
                if line.strip():
                    assignment_records.append(json.loads(line))
    except Exception as e:
        return False, f"Error reading assignment file: {e}"
    
    if not assignment_records:
        return False, "No records found in assignment file"
    
    # Read existing scored qids to avoid duplicates
    scored_qids = set()
    try:
        with open(score_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if 'qid' in record and record['qid'] != 'all':  # Skip global metrics line
                        scored_qids.add(record['qid'])
    except FileNotFoundError:
        pass
    except Exception as e:
        return False, f"Error reading existing score file: {e}"
    
    # Calculate metrics for new records only
    new_metrics = []
    for record in assignment_records:
        qid = record.get('qid')
        if qid and qid not in scored_qids:
            try:
                metrics = calculate_nugget_scores(qid, record['nuggets'])
                metric_record = {
                    'qid': metrics.qid,
                    'strict_vital_score': metrics.strict_vital_score,
                    'strict_all_score': metrics.strict_all_score,
                    'vital_score': metrics.vital_score,
                    'all_score': metrics.all_score,
                }
                new_metrics.append(metric_record)
                scored_qids.add(qid)
            except Exception as e:
                return False, f"Error calculating metrics for qid {qid}: {e}"
    
    # Append new metrics to score file
    if new_metrics:
        try:
            with open(score_file, 'a') as f:
                for metric_record in new_metrics:
                    f.write(json.dumps(metric_record) + '\n')
            
            # If we have all expected lines, calculate and append global metrics
            if len(assignment_records) >= expected_lines:
                # Check if global metrics already exist
                has_global = False
                try:
                    with open(score_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                if record.get('qid') == 'all':
                                    has_global = True
                                    break
                except:
                    pass
                
                if not has_global:
                    from nuggetizer.core.metrics import calculate_global_metrics
                    global_metrics = calculate_global_metrics(assignment_records)
                    with open(score_file, 'a') as f:
                        f.write(json.dumps(global_metrics) + '\n')
            
            return True, f"Saved {len(new_metrics)} new metric records"
        except Exception as e:
            return False, f"Error writing to score file: {e}"
    else:
        return True, "No new metrics to save (all already processed)"

def run_nugget_assignment(rag_results_dir, nugget_file, output_dir, assign_script, score_script, scores_dir, log_file, model="gpt-4o"):
    """Run nugget assignment for all RAG result files in a directory, and score immediately after assignment."""
    rag_results_dir = Path(rag_results_dir)
    nugget_file = Path(nugget_file)
    output_dir = Path(output_dir)
    assign_script = Path(assign_script)
    score_script = Path(score_script)
    scores_dir = Path(scores_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    # Get expected number of lines from nugget file
    with open(nugget_file, 'r') as f:
        expected_lines = sum(1 for _ in f)
    
    rag_result_files = list(rag_results_dir.glob("*.json"))
    log_progress(log_file, f"Found {len(rag_result_files)} RAG result files to process in {rag_results_dir.name}")
    log_progress(log_file, f"Expected {expected_lines} lines per file (from nugget file)")
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm with custom format to show current file
    pbar = tqdm(rag_result_files, desc=f"Assigning Nuggets ({rag_results_dir.name})", unit="file")
    
    for i, rag_file in enumerate(pbar, 1):
        output_assignment_file = output_dir / f"{rag_file.stem}_assignments.jsonl"
        
        # Update progress bar to show current file
        pbar.set_postfix_str(f"Current: {rag_file.name}")
        
        # Only skip if file already exists and has ALL expected lines
        # Otherwise, let assign_nuggets.py handle incremental processing
        if output_assignment_file.exists():
            try:
                with open(output_assignment_file, 'r') as f:
                    lines = [line for line in f if line.strip()]
                    line_count = len(lines)
                if line_count >= expected_lines:
                    log_progress(log_file, f"⏭️  [{i}/{len(rag_result_files)}] Skipping {rag_file.name} (fully processed: {line_count}/{expected_lines} lines)")
                    print(f"⏭️  [{i}/{len(rag_result_files)}] Skipping {rag_file.name} (fully processed: {line_count}/{expected_lines} lines)")
                    successful += 1
                    skipped += 1
                    continue
                else:
                    log_progress(log_file, f"🔄 [{i}/{len(rag_result_files)}] Continuing {rag_file.name} (partially processed: {line_count}/{expected_lines} lines, will process remaining)")
                    print(f"🔄 [{i}/{len(rag_result_files)}] Continuing {rag_file.name} (partially processed: {line_count}/{expected_lines} lines, will process remaining)")
            except Exception as e:
                log_progress(log_file, f"Error checking {output_assignment_file}: {e}, will re-process")
                # Don't delete, let assign_nuggets.py handle it
        
        cmd = [
            "python3", str(assign_script),
            "--nugget_file", str(nugget_file),
            "--answer_file", str(rag_file),
            "--output_file", str(output_assignment_file),
            "--model", model
        ]
        
        try:
            current_msg = f"🔄 [{i}/{len(rag_result_files)}] Processing {rag_file.name}..."
            log_progress(log_file, current_msg)
            print(current_msg)
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            # Verify the output file was created and check progress
            if output_assignment_file.exists():
                with open(output_assignment_file, 'r') as f:
                    lines = [line for line in f if line.strip()]
                    line_count = len(lines)
                if line_count >= expected_lines:
                    success_msg = f"✅ [{i}/{len(rag_result_files)}] Successfully assigned nuggets for {rag_file.name} ({line_count}/{expected_lines} lines - COMPLETE)"
                    log_progress(log_file, success_msg)
                    print(success_msg)
                    successful += 1
                    
                    # Immediately score the assignment file
                    base_name = rag_file.stem
                    score_file = scores_dir / f"{base_name}_scores.jsonl"
                    
                    # Calculate and save metrics incrementally (after each line/query)
                    try:
                        score_msg = f"📊 [{i}/{len(rag_result_files)}] Calculating metrics for {rag_file.name}..."
                        log_progress(log_file, score_msg)
                        print(score_msg)
                        
                        success, message = calculate_and_append_metrics(
                            output_assignment_file, score_file, expected_lines
                        )
                        
                        if success:
                            # Count how many metric records we have
                            try:
                                with open(score_file, 'r') as sf:
                                    score_lines = [line for line in sf if line.strip()]
                                    score_count = len(score_lines)
                                score_success_msg = f"✅ [{i}/{len(rag_result_files)}] Metrics saved for {rag_file.name} ({score_count} metric records) - {message}"
                                log_progress(log_file, score_success_msg)
                                print(score_success_msg)
                            except:
                                score_success_msg = f"✅ [{i}/{len(rag_result_files)}] Metrics saved for {rag_file.name} - {message}"
                                log_progress(log_file, score_success_msg)
                                print(score_success_msg)
                        else:
                            score_error_msg = f"⚠️  [{i}/{len(rag_result_files)}] Failed to calculate metrics for {rag_file.name}: {message}"
                            log_progress(log_file, score_error_msg)
                            print(score_error_msg)
                    except Exception as e:
                        score_error_msg = f"⚠️  [{i}/{len(rag_result_files)}] Exception calculating metrics for {rag_file.name}: {e}"
                        log_progress(log_file, score_error_msg)
                        print(score_error_msg)
                        
                elif line_count > 0:
                    success_msg = f"✅ [{i}/{len(rag_result_files)}] Progress on {rag_file.name} ({line_count}/{expected_lines} lines - PARTIAL, will continue next run)"
                    log_progress(log_file, success_msg)
                    print(success_msg)
                    successful += 1  # Count as success since assign_nuggets.py handles incremental processing
                    
                    # Score partial results if we have enough lines (at least 1)
                    base_name = rag_file.stem
                    score_file = scores_dir / f"{base_name}_scores.jsonl"
                    
                    # Calculate and save metrics for partial results (after each line/query)
                    if line_count > 0:
                        try:
                            score_msg = f"📊 [{i}/{len(rag_result_files)}] Calculating metrics for partial results ({line_count} lines)..."
                            log_progress(log_file, score_msg)
                            print(score_msg)
                            
                            success, message = calculate_and_append_metrics(
                                output_assignment_file, score_file, expected_lines
                            )
                            
                            if success:
                                # Count how many metric records we have
                                try:
                                    with open(score_file, 'r') as sf:
                                        score_lines = [line for line in sf if line.strip()]
                                        score_count = len(score_lines)
                                    score_success_msg = f"✅ [{i}/{len(rag_result_files)}] Metrics saved for partial results ({score_count} metric records) - {message}"
                                    log_progress(log_file, score_success_msg)
                                    print(score_success_msg)
                                except:
                                    score_success_msg = f"✅ [{i}/{len(rag_result_files)}] Metrics saved for partial results - {message}"
                                    log_progress(log_file, score_success_msg)
                                    print(score_success_msg)
                            else:
                                score_error_msg = f"⚠️  [{i}/{len(rag_result_files)}] Failed to calculate metrics: {message}"
                                log_progress(log_file, score_error_msg)
                                print(score_error_msg)
                        except Exception as e:
                            score_error_msg = f"⚠️  [{i}/{len(rag_result_files)}] Exception calculating metrics: {e}"
                            log_progress(log_file, score_error_msg)
                            print(score_error_msg)
                else:
                    error_msg = f"❌ [{i}/{len(rag_result_files)}] Error: {rag_file.name} has 0 lines"
                    log_progress(log_file, error_msg)
                    print(error_msg)
                    failed += 1
            else:
                error_msg = f"❌ [{i}/{len(rag_result_files)}] Error: Output file not created for {rag_file.name}"
                log_progress(log_file, error_msg)
                print(error_msg)
                failed += 1
                
        except subprocess.TimeoutExpired:
            error_msg = f"⏰ [{i}/{len(rag_result_files)}] Timeout processing {rag_file.name}"
            log_progress(log_file, error_msg)
            print(error_msg)
            failed += 1
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ [{i}/{len(rag_result_files)}] Error assigning nuggets for {rag_file.name}: {e.stderr[:200]}"
            log_progress(log_file, error_msg)
            print(error_msg)
            failed += 1
        except Exception as e:
            error_msg = f"❌ [{i}/{len(rag_result_files)}] Unexpected error for {rag_file.name}: {e}"
            log_progress(log_file, error_msg)
            print(error_msg)
            failed += 1
    
    pbar.close()
    
    log_progress(log_file, f"\n🎉 Nugget assignment and scoring complete for {rag_results_dir.name}!")
    log_progress(log_file, f"   ✅ Successful: {successful} (including {skipped} already processed)")
    log_progress(log_file, f"   ⏭️  Skipped (already done): {skipped}")
    log_progress(log_file, f"   ❌ Failed: {failed}")
    log_progress(log_file, f"   📊 Scores calculated immediately after assignment")
    
    return successful, failed

def run_scoring(assignments_dir, scores_dir, script_path, log_file):
    """Run scoring for assignment files."""
    assignments_dir = Path(assignments_dir)
    scores_dir = Path(scores_dir)
    script_path = Path(script_path)
    
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    assignment_files = list(assignments_dir.glob("*_assignments.jsonl"))
    log_progress(log_file, f"Found {len(assignment_files)} assignment files to score")
    
    successful = 0
    failed = 0
    
    pbar = tqdm(assignment_files, desc="Scoring Files", unit="file")
    
    for i, assignment_file in enumerate(pbar, 1):
        # Extract base name (remove _assignments suffix)
        base_name = assignment_file.stem.replace("_assignments", "")
        score_file = scores_dir / f"{base_name}_scores.jsonl"
        
        # Update progress bar to show current file
        pbar.set_postfix_str(f"Current: {assignment_file.name}")
        
        # Skip if already processed
        if score_file.exists():
            skip_msg = f"⏭️  [{i}/{len(assignment_files)}] Skipping {assignment_file.name} (scores already exist)"
            log_progress(log_file, skip_msg)
            print(skip_msg)
            successful += 1
            continue
        
        cmd = [
            "python3", str(script_path),
            "--input_file", str(assignment_file),
            "--output_file", str(score_file)
        ]
        
        try:
            current_msg = f"🔄 [{i}/{len(assignment_files)}] Scoring {assignment_file.name}"
            log_progress(log_file, current_msg)
            print(current_msg)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                success_msg = f"✅ [{i}/{len(assignment_files)}] Successfully scored {assignment_file.name}"
                log_progress(log_file, success_msg)
                print(success_msg)
                successful += 1
            else:
                error_msg = f"❌ [{i}/{len(assignment_files)}] Failed to score {assignment_file.name}"
                log_progress(log_file, error_msg)
                log_progress(log_file, f"   Error: {result.stderr[:200]}")
                print(error_msg)
                failed += 1
                
        except subprocess.TimeoutExpired:
            error_msg = f"⏰ [{i}/{len(assignment_files)}] Timeout scoring {assignment_file.name}"
            log_progress(log_file, error_msg)
            print(error_msg)
            failed += 1
        except Exception as e:
            error_msg = f"💥 [{i}/{len(assignment_files)}] Exception scoring {assignment_file.name}: {e}"
            log_progress(log_file, error_msg)
            print(error_msg)
            failed += 1
    
    pbar.close()
    log_progress(log_file, f"\n🎉 Scoring complete!")
    log_progress(log_file, f"   ✅ Successful: {successful}")
    log_progress(log_file, f"   ❌ Failed: {failed}")
    
    return successful, failed

def main():
    """Main function to run nuggetizer assignment and scoring for RAG results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run nuggetizer assignment and scoring for RAG results')
    parser.add_argument('--rag-results-dir', type=str, 
                       default='/future/u/negara/home/set_based_QPP/querygym/rag_results',
                       help='Base directory containing retrieval and retrieval_cohere folders')
    parser.add_argument('--nugget-file', type=str, 
                       default='/future/u/negara/home/set_based_QPP/data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl',
                       help='Path to nugget file')
    parser.add_argument('--output-dir', type=str,
                       default='/future/u/negara/home/set_based_QPP/querygym/rag_nuggetized_eval',
                       help='Output directory for assignments and scores')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: auto-generated)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='Model to use for nugget assignment (default: gpt-4o)')
    
    args = parser.parse_args()
    
    # Create log file with timestamp if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"/future/u/negara/home/set_based_QPP/rag_nuggetizer_log_{timestamp}.txt"
    else:
        log_file = args.log_file
    
    log_progress(log_file, "="*80)
    log_progress(log_file, "Starting nuggetizer assignment and scoring pipeline for RAG results...")
    log_progress(log_file, "="*80)
    
    # Define paths
    rag_results_base = Path(args.rag_results_dir)
    nugget_file = Path(args.nugget_file)
    output_dir = Path(args.output_dir)
    assign_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/assign_nuggets.py")
    score_script = Path("/future/u/negara/home/set_based_QPP/nuggetizer/scripts/calculate_metrics.py")
    
    # Verify input files exist
    if not rag_results_base.exists():
        log_progress(log_file, f"❌ Error: RAG results base directory not found: {rag_results_base}")
        return False
    
    retrieval_dir = rag_results_base / "retrieval"
    retrieval_cohere_dir = rag_results_base / "retrieval_cohere"
    
    if not retrieval_dir.exists() and not retrieval_cohere_dir.exists():
        log_progress(log_file, f"❌ Error: Neither retrieval nor retrieval_cohere directories found")
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
    log_progress(log_file, f"📁 RAG Results Base: {rag_results_base}")
    log_progress(log_file, f"📁 Nugget File: {nugget_file}")
    log_progress(log_file, f"📁 Output Directory: {output_dir}")
    log_progress(log_file, f"🤖 Model for assignment: {args.model}")
    
    total_assignment_success = 0
    total_assignment_failed = 0
    total_scoring_success = 0
    total_scoring_failed = 0
    
    # Process retrieval folder
    if retrieval_dir.exists():
        assignments_dir = output_dir / "retrieval" / "assignments"
        scores_dir = output_dir / "retrieval" / "scores"
        
        log_progress(log_file, f"\n{'='*80}")
        log_progress(log_file, "PROCESSING RETRIEVAL FOLDER")
        log_progress(log_file, f"{'='*80}")
        
        assignment_success, assignment_failed = run_nugget_assignment(
            retrieval_dir, nugget_file, assignments_dir, assign_script, score_script, scores_dir, log_file, model=args.model
        )
        total_assignment_success += assignment_success
        total_assignment_failed += assignment_failed
        
        # Run catch-up scoring for any files that might have been missed
        if assignment_success > 0:
            log_progress(log_file, f"\n{'='*80}")
            log_progress(log_file, "CATCH-UP SCORING FOR RETRIEVAL (checking for any missed files)")
            log_progress(log_file, f"{'='*80}")
            
            scoring_success, scoring_failed = run_scoring(
                assignments_dir, scores_dir, score_script, log_file
            )
            total_scoring_success += scoring_success
            total_scoring_failed += scoring_failed
        else:
            log_progress(log_file, "❌ Skipping catch-up scoring for retrieval due to assignment failures")
    
    # Process retrieval_cohere folder
    if retrieval_cohere_dir.exists():
        assignments_dir = output_dir / "retrieval_cohere" / "assignments"
        scores_dir = output_dir / "retrieval_cohere" / "scores"
        
        log_progress(log_file, f"\n{'='*80}")
        log_progress(log_file, "PROCESSING RETRIEVAL_COHERE FOLDER")
        log_progress(log_file, f"{'='*80}")
        
        assignment_success, assignment_failed = run_nugget_assignment(
            retrieval_cohere_dir, nugget_file, assignments_dir, assign_script, score_script, scores_dir, log_file, model=args.model
        )
        total_assignment_success += assignment_success
        total_assignment_failed += assignment_failed
        
        # Run catch-up scoring for any files that might have been missed
        if assignment_success > 0:
            log_progress(log_file, f"\n{'='*80}")
            log_progress(log_file, "CATCH-UP SCORING FOR RETRIEVAL_COHERE (checking for any missed files)")
            log_progress(log_file, f"{'='*80}")
            
            scoring_success, scoring_failed = run_scoring(
                assignments_dir, scores_dir, score_script, log_file
            )
            total_scoring_success += scoring_success
            total_scoring_failed += scoring_failed
        else:
            log_progress(log_file, "❌ Skipping catch-up scoring for retrieval_cohere due to assignment failures")
    
    # Final Summary
    log_progress(log_file, f"\n{'='*80}")
    log_progress(log_file, "FINAL SUMMARY")
    log_progress(log_file, f"{'='*80}")
    
    log_progress(log_file, f"Assignment & Inline Scoring: ✅ {total_assignment_success} successful, ❌ {total_assignment_failed} failed")
    log_progress(log_file, f"Catch-up Scoring: ✅ {total_scoring_success} successful, ❌ {total_scoring_failed} failed")
    log_progress(log_file, f"📁 Output directory: {output_dir}")
    log_progress(log_file, f"📝 Complete log saved to: {log_file}")
    
    if total_assignment_failed == 0 and total_scoring_failed == 0:
        log_progress(log_file, f"\n🎉 ALL PROCESSES COMPLETED SUCCESSFULLY!")
        return True
    else:
        log_progress(log_file, f"\n⚠️  Some processes failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

