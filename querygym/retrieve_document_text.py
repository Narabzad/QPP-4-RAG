#!/usr/bin/env python3
"""
Utility script to retrieve document text from TREC RAG index (msmarco-v2.1-doc-segmented).
Given a document ID, returns the full text of the document.
"""
import os
import sys
import json
from pathlib import Path

# Setup Java environment for Pyserini
def setup_java_environment():
    """Set up Java environment variables needed for pyserini."""
    java_home = "/future/u/negara/miniconda3"
    os.environ["JAVA_HOME"] = java_home
    
    jvm_path = "/future/u/negara/miniconda3/lib/server/libjvm.so"
    if os.path.exists(jvm_path):
        os.environ["JVM_PATH"] = jvm_path

setup_java_environment()

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    print("❌ ERROR: Pyserini not found!")
    print("   Please activate pyserini-env: conda activate pyserini-env")
    sys.exit(1)

class DocumentRetriever:
    """Class to retrieve document text from TREC RAG index."""
    
    def __init__(self, index_name='msmarco-v2.1-doc-segmented'):
        """
        Initialize the document retriever.
        
        Args:
            index_name: Name of the prebuilt index to use
        """
        print(f"🔍 Initializing Pyserini searcher for {index_name}...")
        print("⚠️  This may take a few minutes for the first run...")
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        print("✅ Searcher initialized successfully!\n")
    
    def get_document_text(self, docid: str) -> dict:
        """
        Retrieve the text of a document given its ID.
        
        Args:
            docid: Document ID (e.g., "msmarco_v2.1_doc_13_1647729865#0_3617397938")
        
        Returns:
            dict with keys: 'docid', 'title', 'segment', 'full_text', 'success', 'error'
        """
        try:
            doc = self.searcher.doc(docid)
            
            if doc and doc.raw():
                doc_data = json.loads(doc.raw())
                title = doc_data.get('title', '')
                segment = doc_data.get('segment', '')
                full_text = title + " " + segment if title else segment
                
                return {
                    'docid': docid,
                    'title': title,
                    'segment': segment,
                    'full_text': full_text.strip(),
                    'success': True,
                    'error': None
                }
            else:
                return {
                    'docid': docid,
                    'title': '',
                    'segment': '',
                    'full_text': '',
                    'success': False,
                    'error': 'Document not found or empty'
                }
        except Exception as e:
            return {
                'docid': docid,
                'title': '',
                'segment': '',
                'full_text': '',
                'success': False,
                'error': str(e)
            }
    
    def get_documents_batch(self, docids: list) -> list:
        """
        Retrieve text for multiple documents.
        
        Args:
            docids: List of document IDs
        
        Returns:
            List of result dictionaries
        """
        results = []
        for docid in docids:
            result = self.get_document_text(docid)
            results.append(result)
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Retrieve document text from TREC RAG index'
    )
    parser.add_argument(
        'docids',
        nargs='+',
        help='Document ID(s) to retrieve'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output JSON file (optional)'
    )
    parser.add_argument(
        '--index',
        default='msmarco-v2.1-doc-segmented',
        help='Index name (default: msmarco-v2.1-doc-segmented)'
    )
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = DocumentRetriever(index_name=args.index)
    
    # Retrieve documents
    print(f"📄 Retrieving {len(args.docids)} document(s)...\n")
    results = retriever.get_documents_batch(args.docids)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"{'=' * 80}")
        print(f"Document {i}/{len(results)}: {result['docid']}")
        print(f"{'=' * 80}")
        
        if result['success']:
            print(f"✅ Success")
            if result['title']:
                print(f"\n📌 Title: {result['title']}")
            print(f"\n📝 Text ({len(result['full_text'])} characters):")
            print(f"{result['full_text'][:500]}...")  # Show first 500 chars
            print()
        else:
            print(f"❌ Failed: {result['error']}\n")
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # python retrieve_document_text.py "msmarco_v2.1_doc_13_1647729865#0_3617397938"
    # python retrieve_document_text.py "docid1" "docid2" "docid3" -o output.json
    
    main()



