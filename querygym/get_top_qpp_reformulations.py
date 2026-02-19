#!/usr/bin/env python3
"""
Script to extract top 5 highest QPP reformulations for each query and QPP method.

For each query (qid) and each QPP method, this script:
1. Collects all reformulations with their QPP scores
2. Sorts by QPP score (descending)
3. Selects the top 5 reformulations
4. Stores the results in a JSON file

Output structure:
{
  "qid": {
    "qpp_method": [
      {
        "method": "genqr",
        "trial": 1,
        "reformulated_query": "...",
        "qpp_score": 123.45,
        "rank": 1
      },
      ...
    ]
  }
}
"""

import json
import argparse
import math
from collections import defaultdict
from pathlib import Path


def get_all_qpp_methods(data):
    """Extract all unique QPP method names from the data."""
    qpp_methods = set()
    
    for query_id, query_data in data.items():
        for reformulation in query_data.get('reformulations', []):
            # Pre-retrieval QPP metrics
            pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
            qpp_methods.update(pre_qpp.keys())
            
            # Post-retrieval QPP metrics (pyserini and cohere)
            qpp_metrics_dict = reformulation.get('qpp_metrics', {})
            for retrieval_method in ['pyserini', 'cohere']:
                retrieval_qpp = qpp_metrics_dict.get(retrieval_method, {})
                for metric_name in retrieval_qpp.keys():
                    # Prefix with retrieval method to distinguish
                    qpp_methods.add(f"{retrieval_method}_{metric_name}")
    
    return sorted(list(qpp_methods))


def get_qpp_score(reformulation, qpp_method):
    """Get QPP score for a specific method from a reformulation."""
    # Check if it's a post-retrieval metric (has retrieval method prefix)
    if '_' in qpp_method:
        retrieval_method, metric_name = qpp_method.split('_', 1)
        qpp_metrics_dict = reformulation.get('qpp_metrics', {})
        retrieval_qpp = qpp_metrics_dict.get(retrieval_method, {})
        score = retrieval_qpp.get(metric_name)
    else:
        # Pre-retrieval metric
        pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
        score = pre_qpp.get(qpp_method)
    
    # Convert to float if possible, return None if invalid
    if score is None:
        return None
    try:
        return float(score)
    except (ValueError, TypeError):
        return None


def extract_top_qpp_reformulations(data, top_k=5):
    """
    Extract top K reformulations for each query and QPP method.
    
    Args:
        data: Consolidated query data dictionary
        top_k: Number of top reformulations to extract (default: 5)
    
    Returns:
        Dictionary with structure: qid -> qpp_method -> list of top K reformulations
    """
    # Get all QPP methods
    all_qpp_methods = get_all_qpp_methods(data)
    print(f"Found {len(all_qpp_methods)} QPP methods")
    
    results = {}
    
    # Process each query
    for qid, query_data in data.items():
        print(f"Processing query: {qid}")
        results[qid] = {}
        
        # For each QPP method, collect all reformulations with scores
        for qpp_method in all_qpp_methods:
            reformulation_scores = []
            
            for reformulation in query_data.get('reformulations', []):
                qpp_score = get_qpp_score(reformulation, qpp_method)
                
                # Skip if score is None, NaN, or invalid
                if qpp_score is None:
                    continue
                try:
                    # Check if it's a valid number (not NaN or inf)
                    if math.isnan(qpp_score) or math.isinf(qpp_score):
                        continue
                except (TypeError, ValueError):
                    continue
                
                # Create entry for this reformulation
                entry = {
                    'method': reformulation.get('method'),
                    'trial': reformulation.get('trial'),
                    'reformulated_query': reformulation.get('reformulated_query', ''),
                    'qpp_score': qpp_score
                }
                reformulation_scores.append(entry)
            
            # Sort by QPP score (descending) and take top K
            if reformulation_scores:
                reformulation_scores.sort(key=lambda x: x['qpp_score'], reverse=True)
                top_reformulations = reformulation_scores[:top_k]
                
                # Add rank to each entry
                for rank, entry in enumerate(top_reformulations, start=1):
                    entry['rank'] = rank
                
                results[qid][qpp_method] = top_reformulations
            else:
                # No valid scores found for this QPP method
                results[qid][qpp_method] = []
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract top K highest QPP reformulations for each query and QPP method'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='consolidated_query_data.json',
        help='Path to consolidated query data JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='top_qpp_reformulations.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top reformulations to extract (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Load consolidated data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} queries")
    
    # Extract top reformulations
    print(f"\nExtracting top {args.top_k} reformulations for each QPP method...")
    results = extract_top_qpp_reformulations(data, top_k=args.top_k)
    
    # Save results
    output_path = Path(args.output)
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    total_queries = len(results)
    qpp_methods_count = defaultdict(int)
    reformulations_count = defaultdict(int)
    
    for qid, qpp_data in results.items():
        for qpp_method, reformulations in qpp_data.items():
            if reformulations:
                qpp_methods_count[qpp_method] += 1
                reformulations_count[qid] += len(reformulations)
    
    print(f"Total queries processed: {total_queries}")
    print(f"Total QPP methods: {len(qpp_methods_count)}")
    print(f"Average reformulations per query: {sum(reformulations_count.values()) / total_queries:.2f}")
    
    # Show sample output for first query
    if results:
        first_qid = list(results.keys())[0]
        first_qpp_method = list(results[first_qid].keys())[0]
        print(f"\nSample output for query {first_qid}, QPP method '{first_qpp_method}':")
        sample = results[first_qid][first_qpp_method]
        if sample:
            print(f"  Top reformulation:")
            print(f"    Method: {sample[0]['method']}, Trial: {sample[0]['trial']}")
            print(f"    QPP Score: {sample[0]['qpp_score']}")
            print(f"    Query: {sample[0]['reformulated_query'][:100]}...")
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
