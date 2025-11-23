#!/usr/bin/env python3
"""
Demo script for Week 5 Hybrid Retrieval System

This script demonstrates:
1. Downloading arXiv papers (optional)
2. Building a hybrid index from PDFs
3. Performing vector, keyword, and hybrid searches
4. Comparing results from all three methods
5. Evaluating performance
"""

from rag_pipeline import ArXivRAGPipeline, download_arxiv_papers
from pathlib import Path
import json

def print_results(results, title, k=3):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)
    
    for i, r in enumerate(results[:k], 1):
        print(f"\nRank {i}:")
        print(f"  Paper: {r.get('paper', r.get('metadata', {}).get('paper', 'Unknown'))}")
        
        # Show different scores based on what's available
        if 'hybrid_score' in r:
            print(f"  Hybrid Score: {r['hybrid_score']:.4f}")
            if 'vector_score' in r:
                print(f"    ├─ Vector: {r['vector_score']:.4f}")
            if 'keyword_score' in r:
                print(f"    └─ Keyword: {r['keyword_score']:.4f}")
        elif 'rrf_score' in r:
            print(f"  RRF Score: {r['rrf_score']:.4f}")
        elif 'distance' in r:
            print(f"  Distance: {r['distance']:.4f}")
        elif 'score' in r:
            print(f"  BM25 Score: {r['score']:.4f}")
        
        chunk = r.get('chunk', '')
        print(f"  Preview: {chunk[:200]}...")


def download_papers(num_papers=50):
    """Download arXiv papers for the demo"""
    print("\n" + "="*80)
    print("DOWNLOADING ARXIV PAPERS")
    print("="*80)
    
    pdf_dir = Path('arxiv_papers')
    
    # Check if we already have papers
    if pdf_dir.exists():
        existing_pdfs = list(pdf_dir.glob('*.pdf'))
        if len(existing_pdfs) >= num_papers:
            print(f"\n✓ Found {len(existing_pdfs)} existing PDFs in '{pdf_dir}'")
            print("  Skipping download. Delete the folder to re-download.")
            return list(existing_pdfs)
    
    print(f"\nDownloading {num_papers} papers from arXiv (cs.CL category)...")
    print("This may take a few minutes...\n")
    
    try:
        pdf_paths = download_arxiv_papers(
            category='cs.CL',
            max_results=num_papers,
            output_dir='arxiv_papers'
        )
        print(f"\n✓ Successfully downloaded {len(pdf_paths)} papers!")
        return pdf_paths
    except Exception as e:
        print(f"\n❌ Error downloading papers: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have 'arxiv' package installed:")
        print("   pip install --break-system-packages arxiv")
        print("2. Check your internet connection")
        print("3. Try reducing num_papers if rate limited")
        return []


def build_index(pipeline, pdf_paths):
    """Build the hybrid index from PDFs"""
    print("\n" + "="*80)
    print("BUILDING HYBRID INDEX")
    print("="*80)
    
    if not pdf_paths:
        print("\n❌ No PDF files provided!")
        return False
    
    print(f"\nProcessing {len(pdf_paths)} papers...")
    print("This will:")
    print("  1. Extract text from PDFs")
    print("  2. Chunk text into segments")
    print("  3. Generate embeddings")
    print("  4. Build FAISS index")
    print("  5. Create SQLite database with FTS5")
    print("\nThis may take several minutes...\n")
    
    try:
        pipeline.process_papers(pdf_paths, chunk_size=512, overlap=50)
        
        # Save index
        pipeline.save_index('faiss_index.bin', 'chunks.json', 'metadata.json')
        
        print("\n✓ Index built and saved successfully!")
        print(f"  - Total chunks: {len(pipeline.chunks)}")
        print(f"  - Total papers: {len(set(m['paper'] for m in pipeline.metadata))}")
        print(f"  - Database: {pipeline.db_path}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        return False


def main():
    print("="*80)
    print("WEEK 5 HYBRID RETRIEVAL SYSTEM - DEMO")
    print("="*80)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = ArXivRAGPipeline(
        model_name='all-MiniLM-L6-v2',
        db_path='arxiv_hybrid_demo.db'
    )
    print("   ✓ Pipeline initialized")
    
    # Try to load existing index
    index_loaded = False
    try:
        print("\n2. Checking for existing index...")
        pipeline.load_index(
            'faiss_index.bin',
            'chunks.json',
            'metadata.json',
            db_path='arxiv_hybrid_demo.db'
        )
        print("   ✓ Index loaded successfully!")
        print(f"   - Total chunks: {len(pipeline.chunks)}")
        print(f"   - Total papers: {len(set(m['paper'] for m in pipeline.metadata))}")
        index_loaded = True
        
    except Exception as e:
        print(f"   ⚠ No existing index found")
    
    # If no index exists, download and build
    if not index_loaded:
        print("\n" + "="*80)
        print("No index found. Let's build one!")
        print("="*80)
        
        # Ask user if they want to download papers
        print("\nOptions:")
        print("  1. Download papers from arXiv (recommended for demo)")
        print("  2. Use PDFs from 'arxiv_papers' directory")
        print("  3. Exit and add PDFs manually")
        
        choice = input("\nEnter choice (1-3) [1]: ").strip() or "1"
        
        if choice == "1":
            # Download papers
            num_papers = input("\nHow many papers to download? [10]: ").strip() or "10"
            try:
                num_papers = int(num_papers)
                num_papers = max(1, min(num_papers, 50))  # Limit 1-50
            except:
                num_papers = 10
            
            pdf_paths = download_papers(num_papers)
            if not pdf_paths:
                print("\n❌ Could not download papers. Exiting.")
                return
            
            # Build index
            if not build_index(pipeline, pdf_paths):
                print("\n❌ Could not build index. Exiting.")
                return
            
        elif choice == "2":
            # Use existing PDFs
            pdf_dir = Path('arxiv_papers')
            if not pdf_dir.exists():
                print(f"\n❌ Directory '{pdf_dir}' not found!")
                print(f"Please create it and add PDF files, then run again.")
                return
            
            pdf_paths = list(pdf_dir.glob('*.pdf'))
            if not pdf_paths:
                print(f"\n❌ No PDF files found in '{pdf_dir}'!")
                print("Please add some PDF files and try again.")
                return
            
            print(f"\n✓ Found {len(pdf_paths)} PDF files")
            
            # Build index
            if not build_index(pipeline, pdf_paths):
                print("\n❌ Could not build index. Exiting.")
                return
        
        else:
            print("\nExiting. Add PDFs to 'arxiv_papers' directory and run again.")
            return
    
    # Demo queries
    demo_queries = [
        "transformer attention mechanism",
        "neural networks deep learning",
        "machine learning algorithms"
    ]
    
    print("\n" + "="*80)
    print("3. RUNNING DEMO QUERIES")
    print("="*80)
    
    for query in demo_queries:
        print(f"\n\n{'#'*80}")
        print(f"Query: '{query}'")
        print('#'*80)
        
        # Vector search
        print("\n[1/4] Vector Search (FAISS - Semantic Similarity)")
        vector_results = pipeline.search(query, k=3, method='vector')
        print_results(vector_results, "Vector Search Results", k=3)
        
        # Keyword search
        print("\n\n[2/4] Keyword Search (SQLite FTS5 - BM25)")
        keyword_results = pipeline.search(query, k=3, method='keyword')
        if keyword_results:
            print_results(keyword_results, "Keyword Search Results", k=3)
        else:
            print("\n   No keyword matches found (try different query terms)")
        
        # Hybrid search (weighted)
        print("\n\n[3/4] Hybrid Search (Weighted Fusion, α=0.5)")
        hybrid_results = pipeline.search(
            query, k=3, method='hybrid', 
            alpha=0.5, fusion_method='weighted'
        )
        print_results(hybrid_results, "Hybrid Search Results (Weighted)", k=3)
        
        # Hybrid search (RRF)
        print("\n\n[4/4] Hybrid Search (Reciprocal Rank Fusion)")
        rrf_results = pipeline.search(
            query, k=3, method='hybrid',
            fusion_method='rrf'
        )
        print_results(rrf_results, "Hybrid Search Results (RRF)", k=3)
    
    # Evaluation (if test queries are defined)
    print("\n\n" + "="*80)
    print("4. EVALUATION (Sample)")
    print("="*80)
    
    # Create sample test queries
    # NOTE: You should replace these with actual paper names from your index
    test_queries = [
        {
            'query': 'transformer attention',
            'relevant_docs': [m['paper'] for m in pipeline.metadata[:2]]
        },
        {
            'query': 'neural network',
            'relevant_docs': [m['paper'] for m in pipeline.metadata[2:4]]
        },
    ]
    
    print(f"\nRunning evaluation on {len(test_queries)} sample queries...")
    metrics = pipeline.evaluate(test_queries, k=3)
    
    print("\n" + "-"*80)
    print("Recall Results:")
    print("-"*80)
    for method, score in metrics.items():
        print(f"  {method:25s}: {score:.3f} ({score*100:.1f}%)")
    
    # Summary
    print("\n\n" + "="*80)
    print("5. SUMMARY")
    print("="*80)
    
    print(f"""
    ✓ Hybrid retrieval system successfully demonstrated!
    
    Key Features:
    - Vector Search (FAISS): Semantic similarity matching
    - Keyword Search (FTS5): Exact term matching with BM25
    - Hybrid Search: Combines both methods for best results
    - Two fusion strategies: Weighted Sum & Reciprocal Rank Fusion
    
    Index Statistics:
    - Total chunks: {len(pipeline.chunks)}
    - Total papers: {len(set(m['paper'] for m in pipeline.metadata))}
    - Embedding dimension: {pipeline.embedding_dim}
    - Database: {pipeline.db_path}
    
    Next Steps:
    1. Test with more queries
    2. Tune alpha parameter (0.0 to 1.0)
    3. Try different fusion methods
    4. Run full evaluation (Week5_Evaluation.ipynb)
    5. Deploy with FastAPI (fastapi_service_hybrid.py)
    """)
    
    print("="*80)
    print("Demo completed! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()