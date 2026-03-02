#!/usr/bin/env python3
"""
Standalone script to process a single QPP task.
Called from subprocess to avoid JVM conflicts.
"""

import sys
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_REPO))

# Import the processing functions directly
# Add the qpp directory to path
qpp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, qpp_dir)

# Import from run_qpp_querygym
import importlib.util
spec = importlib.util.spec_from_file_location("run_qpp_querygym", os.path.join(qpp_dir, "run_qpp_querygym.py"))
run_qpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_qpp)

process_pre_retrieval_metrics = run_qpp.process_pre_retrieval_metrics
process_post_retrieval_metrics = run_qpp.process_post_retrieval_metrics

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pre', 'post'], required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--run_file', default=None)
    parser.add_argument('--k_top', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'pre':
            result = process_pre_retrieval_metrics(args.query_file, args.index_path, args.output_dir)
            if result:
                print(f"SUCCESS: {result}")
                sys.exit(0)
            else:
                print(f"FAILED: Pre-retrieval processing failed")
                sys.exit(1)
        else:  # post
            if not args.run_file:
                print("FAILED: --run_file required for post-retrieval mode")
                sys.exit(1)
            result = process_post_retrieval_metrics(
                args.query_file, args.run_file, args.index_path, args.output_dir, args.k_top
            )
            if result:
                print(f"SUCCESS: {result}")
                sys.exit(0)
            else:
                print(f"FAILED: Post-retrieval processing failed")
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
