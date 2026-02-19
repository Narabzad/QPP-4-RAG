#!/usr/bin/env python3
"""
Test different ways to access document vectors from the index
"""

import sys
sys.path.append('/future/u/negara/home/set_based_QPP')

from pyserini.index import LuceneIndexReader

index_path = "msmarco-v2.1-doc-segmented"

print("Loading index...")
index_reader = LuceneIndexReader.from_prebuilt_index(index_path)
print(f"✓ Index loaded. Total docs: {index_reader.stats()['documents']}")

# Test document ID
test_doc_id = "msmarco_v2.1_doc_16_287012450#4_490828734"

print(f"\nTesting document ID: {test_doc_id}")

# Method 1: get_document_vector
print("\n1. Testing get_document_vector()...")
try:
    doc_vector = index_reader.get_document_vector(test_doc_id)
    if doc_vector:
        print(f"   ✓ Success! Vector size: {len(doc_vector)}")
        print(f"   Sample terms: {list(doc_vector.keys())[:10]}")
    else:
        print(f"   ✗ Returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 2: Try without the passage part
base_doc_id = test_doc_id.split('#')[0]
print(f"\n2. Testing with base doc ID: {base_doc_id}")
try:
    doc_vector = index_reader.get_document_vector(base_doc_id)
    if doc_vector:
        print(f"   ✓ Success! Vector size: {len(doc_vector)}")
        print(f"   Sample terms: {list(doc_vector.keys())[:10]}")
    else:
        print(f"   ✗ Returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 3: Try to get raw document
print(f"\n3. Testing doc() method...")
try:
    doc = index_reader.doc(test_doc_id)
    if doc:
        print(f"   ✓ Document found!")
        print(f"   Doc ID: {doc.id()}")
        print(f"   Raw content length: {len(doc.raw())}")
    else:
        print(f"   ✗ Document not found")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 4: Check if we can get term frequencies differently
print(f"\n4. Testing term vector retrieval...")
try:
    # Try to get the internal Lucene doc ID
    lucene_doc_id = index_reader.convert_external_docid_to_internal(test_doc_id)
    print(f"   Lucene internal ID: {lucene_doc_id}")
    
    # Now try to get term vector
    doc_vector = index_reader.get_document_vector(test_doc_id)
    print(f"   Document vector: {doc_vector}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Method 5: Check index configuration
print(f"\n5. Index configuration:")
print(f"   Index type: {index_reader.stats()}")
print(f"   Has positions: {index_reader.object.hasPositions()}")
print(f"   Has offsets: {index_reader.object.hasOffsets()}")
print(f"   Has vectors: {index_reader.object.hasVectors()}")
