#!/usr/bin/env python3
"""Quick test to see how long pre-retrieval actually takes"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_REPO))

from pyserini.index import LuceneIndexReader
import time

# Test with topics.original.txt
query_file = str(_REPO / "querygym/queries/topics.original.txt")
index_path = "msmarco-v2.1-doc-segmented"

print("Loading index...")
start = time.time()
index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
print(f"✓ Index loaded in {time.time()-start:.1f}s")

# Load queries
queries = {}
with open(query_file, 'r') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]

print(f"✓ Loaded {len(queries)} queries")

# Collect unique tokens
print("\nCollecting unique tokens...")
start = time.time()
tokens = set()
for qtext in queries.values():
    qtokens = index_reader.analyze(qtext)
    tokens.update(qtokens)
print(f"✓ Found {len(tokens)} unique tokens in {time.time()-start:.1f}s")

# Test token processing speed
print("\nTesting token processing speed...")
start = time.time()
test_tokens = list(tokens)[:10]  # Test with 10 tokens

for i, token in enumerate(test_tokens):
    t_start = time.time()
    # Simulate VAR calculation
    postings = index_reader.get_postings_list(token, analyzer=None)
    t_elapsed = time.time() - t_start
    
    if postings:
        print(f"  Token {i+1}/10: {token[:20]:20s} - {len([p for p in postings])} docs - {t_elapsed:.2f}s")
    else:
        print(f"  Token {i+1}/10: {token[:20]:20s} - 0 docs - {t_elapsed:.2f}s")

total_time = time.time() - start
avg_time = total_time / 10

print(f"\n✓ Processed 10 tokens in {total_time:.1f}s")
print(f"  Average: {avg_time:.2f}s per token")
print(f"\n📊 ESTIMATES:")
print(f"  Total tokens: {len(tokens)}")
print(f"  Estimated time for all tokens: {len(tokens) * avg_time / 60:.1f} minutes")
print(f"  With 20 parallel workers: {len(tokens) * avg_time / 60 / 20:.1f} minutes per file")
print(f"  For 31 query files: {31 * len(tokens) * avg_time / 60 / 20:.1f} minutes total")
