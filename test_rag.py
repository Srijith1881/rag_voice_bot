#!/usr/bin/env python3
"""
Test script to verify RAG system is working correctly
Run this to test your RAG engine without starting the voice assistant
"""

import os
from dotenv import load_dotenv
from rag_engine import get_rag_chain

# Load environment variables
load_dotenv(".env")

def test_rag():
    print("=" * 60)
    print("🧪 Testing RAG System")
    print("=" * 60)
    
    # Initialize RAG - will load from persistent cache if available
    # If cache doesn't exist, run: python precompute_vectors.py first
    rag_chain = get_rag_chain()
    
    # Test questions
    test_questions = [
        "What are electric cars?",
        "What is the range of electric vehicles?",
        "how many plug-in electric cars were sold in 2023",
        "How do electric cars work?",
        "What is apple",  # This should NOT be in knowledge base
    ]
    
    print("\n" + "=" * 60)
    print("📝 Running Test Questions")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}/{len(test_questions)}")
        print(f"Question: {question}")
        print(f"{'─' * 60}")
        
        try:
            answer = rag_chain.run(question)
            print(f"\n✅ Answer received successfully")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_rag()

