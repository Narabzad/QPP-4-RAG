#!/usr/bin/env python3
"""
Convert JSON array format to JSONL format for nuggetizer.
"""

import json
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

def convert_json_to_jsonl(input_file, output_file):
    """Convert JSON array to JSONL."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    base_dir = _REPO / "querygym"
    input_dir = base_dir / "rag_results_o"
    output_dir = base_dir / "rag_results_o_jsonl"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(input_dir.glob("*.json"))
    
    print(f"📁 Found {len(json_files)} files to convert")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    for json_file in tqdm(json_files, desc="Converting"):
        output_file = output_dir / f"{json_file.stem}.jsonl"
        
        if output_file.exists():
            continue
        
        try:
            convert_json_to_jsonl(json_file, output_file)
        except Exception as e:
            print(f"❌ Error converting {json_file.name}: {e}")
    
    print(f"✅ Conversion complete!")
    print(f"📁 JSONL files saved to: {output_dir}")

if __name__ == "__main__":
    main()
