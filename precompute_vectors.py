#!/usr/bin/env python3
"""
Pre-compute and store vector embeddings for the PDF document.
Run this script once to process the PDF and create persistent vectors.
After this, the voice bot will load vectors from cache (much faster).
"""

import os
from dotenv import load_dotenv
from rag_engine import initialize_rag

# Load environment variables
load_dotenv(".env")

def precompute_vectors():
    """
    Pre-compute vector embeddings and save them to disk.
    This only needs to be run once (or when PDF changes).
    """
    print("=" * 60)
    print("🔧 Pre-computing Vector Embeddings")
    print("=" * 60)
    print("\nThis will process your PDF and create persistent vector embeddings.")
    print("After this completes, the voice bot will load vectors instantly.\n")
    
    # Force creation of new vectors (rebuild even if cache exists)
    print("📝 Processing PDF and creating embeddings...")
    rag_chain = initialize_rag(use_cache=True, force_rebuild=True)
    
    print("\n" + "=" * 60)
    print("✅ Vector embeddings pre-computed and saved!")
    print("=" * 60)
    print(f"\n📦 Vectors stored at: {os.path.abspath('data/faiss_index')}")
    print("🚀 Your voice bot will now load vectors from cache (much faster!)\n")
    
    return rag_chain

if __name__ == "__main__":
    precompute_vectors()

