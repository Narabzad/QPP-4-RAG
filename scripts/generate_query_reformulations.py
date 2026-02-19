#!/usr/bin/env python3
"""
Script to generate query reformulations using GPT-4o.
Reads queries from topics.rag24.test.txt and generates 20 variations for each query.
Each set of 20 reformulations follows 10 specific instructions (2 reformulations per instruction).
"""

import os
import time
from pathlib import Path
from openai import OpenAI

def setup_openai_client():
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv("SEWON_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("SEWON_OPENAI_API_KEY environment variable not found")
    return OpenAI(api_key=api_key)

def read_topics_file(file_path):
    """Read and parse the topics file."""
    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    query_id, query_text = parts
                    topics.append((query_id, query_text))
    return topics

def get_reformulation_prompt(query):
    """Generate the reformulation prompt for GPT-4o."""
    system_prompt = """You are an expert query reformulator for information retrieval. Your goal is to generate a *diverse set* of reformulations that improve recall and coverage while preserving the original information need. Think carefully, but return only the final list in the required format."""
    
    user_prompt = f"""Given the query Q, produce 20 reformulations following these specific instructions. Generate EXACTLY 2 reformulations for each of the 10 instructions below:

1. Improve the search effectiveness by suggesting expansion terms for the query
2. Recommend expansion terms for the query to improve search results
3. Improve the search effectiveness by suggesting useful expansion terms for the query
4. Maximize search utility by suggesting relevant expansion phrases for the query
5. Enhance search efficiency by proposing valuable terms to expand the query
6. Elevate search performance by recommending relevant expansion phrases for the query
7. Boost the search accuracy by providing helpful expansion terms to enrich the query
8. Increase the search efficacy by offering beneficial expansion keywords for the query
9. Optimize search results by suggesting meaningful expansion terms to enhance the query
10. Enhance search outcomes by recommending beneficial expansion terms to supplement the query

Rules:
1) Preserve the original information need; do not add unsupported facts.
2) Each reformulation should follow the specific instruction it's based on.
3) Keep each reformulation concise i.e., < 20 words.
4) No explanations—return only the list.

Output format
Return EXACTLY 20 reformulations, one per line. No numbering, labels, or extra text.

Q = "{query}\""""
    
    return system_prompt, user_prompt

def generate_reformulations(client, query):
    """Generate 20 reformulations for a given query using GPT-4o."""
    system_prompt, user_prompt = get_reformulation_prompt(query)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse line-by-line response (one reformulation per line)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) == 20:
            return lines
        elif len(lines) > 20:
            print(f"Warning: Got {len(lines)} reformulations, taking first 20")
            return lines[:20]
        else:
            print(f"Warning: Expected 20 reformulations, got {len(lines)}")
            # Pad with original query if needed
            while len(lines) < 20:
                lines.append(query)
            return lines
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []

def create_output_files(topics, reformulations_data, output_dir):
    """Create separate output files for each reformulation variant."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize 20 output files
    for i in range(1, 21):
        output_file = output_dir / f"topics.rag24.test.v{i}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for j, (query_id, original_query) in enumerate(topics):
                if j < len(reformulations_data) and reformulations_data[j]:
                    reformulations = reformulations_data[j]
                    if i-1 < len(reformulations):
                        reformulated_query = reformulations[i-1]  # Now it's just a string
                        f.write(f"{query_id}\t{reformulated_query}\n")
                    else:
                        # Fallback to original query if not enough reformulations
                        f.write(f"{query_id}\t{original_query}\n")
                else:
                    # Fallback to original query if reformulation failed
                    f.write(f"{query_id}\t{original_query}\n")

def main():
    """Main function to process all queries and generate reformulations."""
    print("Starting query reformulation process...")
    
    # Setup
    client = setup_openai_client()
    input_file = "/future/u/negara/home/RAG-Query/topics.rag24.test.txt"
    output_dir = "/future/u/negara/home/RAG-Query/query_reformulation_t0.1"
    
    # Read topics
    print(f"Reading topics from {input_file}...")
    topics = read_topics_file(input_file)
    print(f"Found {len(topics)} queries to process")
    
    # Process each query
    reformulations_data = []
    for i, (query_id, query_text) in enumerate(topics):
        print(f"Processing query {i+1}/{len(topics)}: {query_id}")
        print(f"Original query: {query_text}")
        
        reformulations = generate_reformulations(client, query_text)
        reformulations_data.append(reformulations)
        
        if reformulations:
            print(f"Generated {len(reformulations)} reformulations")
        else:
            print("Failed to generate reformulations - will use original query")
        
        # Add delay to respect API rate limits
        time.sleep(1)
        
        # Progress update every 10 queries
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(topics)} queries")
    
    # Create output files
    print(f"Creating output files in {output_dir}...")
    create_output_files(topics, reformulations_data, output_dir)
    print("Query reformulation process completed!")
    
    # Summary
    successful_reformulations = sum(1 for r in reformulations_data if r)
    print(f"Summary:")
    print(f"  Total queries processed: {len(topics)}")
    print(f"  Successful reformulations: {successful_reformulations}")
    print(f"  Failed reformulations: {len(topics) - successful_reformulations}")
    print(f"  Output files created: 20 (topics.rag24.test.v1.txt to topics.rag24.test.v20.txt)")

if __name__ == "__main__":
    main()
