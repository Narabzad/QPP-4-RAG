#!/usr/bin/env python3
"""
Generate query files (topics format) from Qgym reformulation results.
Creates 31 files: 6 methods × 5 trials + 1 original queries file.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_reformulation_data(json_file):
    """Load the reformulation results JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_query_files(json_file, output_dir):
    """Generate query files for each method/trial combination and original queries."""
    
    # Load data
    print("📖 Loading reformulation data...")
    data = load_reformulation_data(json_file)
    print(f"✅ Loaded {len(data)} entries")
    
    # Organize data by method and trial
    method_trial_queries = defaultdict(lambda: defaultdict(list))
    original_queries = {}
    
    for entry in data:
        query_id = entry['query_id']
        method_name = entry['method_name']
        trial = entry['trial']
        reformulated_query = entry['reformulated_query']
        original_query = entry['original_query']
        
        # Store reformulated queries by method and trial
        method_trial_queries[method_name][trial].append((query_id, reformulated_query))
        
        # Store original queries (only need to store once per query_id)
        if query_id not in original_queries:
            original_queries[query_id] = original_query
    
    # Get all methods
    methods = sorted(method_trial_queries.keys())
    print(f"📋 Found {len(methods)} methods: {', '.join(methods)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate files for each method and trial
    files_created = []
    
    for method in methods:
        for trial in range(1, 6):  # Trials 1-5
            if trial in method_trial_queries[method]:
                queries = method_trial_queries[method][trial]
                
                # Sort by query_id for consistency
                queries.sort(key=lambda x: x[0])
                
                # Create filename: topics.method_trial.txt
                filename = f"topics.{method}_trial{trial}.txt"
                filepath = output_path / filename
                
                # Write queries in topics format (query_id\tquery_text)
                with open(filepath, 'w', encoding='utf-8') as f:
                    for query_id, query_text in queries:
                        f.write(f"{query_id}\t{query_text}\n")
                
                files_created.append((filename, len(queries)))
                print(f"✅ Created {filename} with {len(queries)} queries")
    
    # Generate original queries file
    original_filename = "topics.original.txt"
    original_filepath = output_path / original_filename
    
    # Sort by query_id
    sorted_original = sorted(original_queries.items(), key=lambda x: x[0])
    
    with open(original_filepath, 'w', encoding='utf-8') as f:
        for query_id, query_text in sorted_original:
            f.write(f"{query_id}\t{query_text}\n")
    
    files_created.append((original_filename, len(sorted_original)))
    print(f"✅ Created {original_filename} with {len(sorted_original)} queries")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total files created: {len(files_created)}")
    print(f"   Output directory: {output_path}")
    
    return files_created

def main():
    import argparse
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Generate query files from Qgym reformulation results')
    parser.add_argument('--input', type=str,
                       default=str(_here.parent / "data" / "Qgym_reformulation_results.json"),
                       help='Path to reformulation results JSON file')
    parser.add_argument('--output-dir', type=str,
                       default=str(_here / "queries"),
                       help='Output directory for generated query files')
    args = parser.parse_args()

    json_file = Path(args.input)
    output_dir = Path(args.output_dir)

    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return

    print("🚀 Generating query files from Qgym reformulation results...")
    print(f"📂 Input: {json_file}")
    print(f"📂 Output: {output_dir}")
    print("=" * 60)

    files_created = generate_query_files(json_file, output_dir)

    print("\n🎉 Query file generation completed!")
    print(f"📁 Created {len(files_created)} query files")

if __name__ == "__main__":
    main()







