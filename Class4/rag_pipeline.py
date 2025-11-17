import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
import pickle
from pathlib import Path
import requests
import time
from tqdm import tqdm


class ArXivRAGPipeline:
    """Complete RAG pipeline for arXiv papers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG pipeline with an embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunks = []
        self.metadata = []
        self.index = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Open a PDF and extract all text as a single string.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Full text extracted from the PDF
        """
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            page_text = page.get_text()  # get raw text from page
            # Basic cleaning: remove excessive whitespace
            page_text = ' '.join(page_text.split())
            pages.append(page_text)
        full_text = "\n\n".join(pages)
        doc.close()
        return full_text
    
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        tokens = text.split()
        chunks = []
        step = max_tokens - overlap
        
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + max_tokens]
            chunk_text = " ".join(chunk)
            if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk_text)
        
        return chunks
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (num_chunks, dim)
            
        Returns:
            FAISS index
        """
        dim = embeddings.shape[1]
        print(f"Building FAISS index with dimension {dim}...")
        
        # Using IndexFlatL2 for exact search (can switch to IndexIVFFlat for larger datasets)
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        print(f"Index built with {index.ntotal} vectors")
        return index
    
    def process_papers(self, pdf_paths: List[str], chunk_size: int = 512, overlap: int = 50):
        """
        Process multiple papers: extract text, chunk, embed, and index.
        
        Args:
            pdf_paths: List of paths to PDF files
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        all_chunks = []
        all_metadata = []
        
        print(f"\nProcessing {len(pdf_paths)} papers...")
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            try:
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                # Chunk text
                chunks = self.chunk_text(text, max_tokens=chunk_size, overlap=overlap)
                
                # Store chunks with metadata
                paper_name = Path(pdf_path).stem
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'paper': paper_name,
                        'chunk_id': chunk_idx,
                        'source_path': pdf_path
                    })
                    
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        # Generate embeddings
        embeddings = self.generate_embeddings(all_chunks)
        
        # Build FAISS index
        self.index = self.build_faiss_index(embeddings)
        
        print(f"\nPipeline ready with {len(self.chunks)} chunks from {len(pdf_paths)} papers")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call process_papers() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'distance': float(dist),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save_index(self, index_path: str, chunks_path: str, metadata_path: str):
        """
        Save the FAISS index and associated data.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks
            metadata_path: Path to save metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved index to {index_path}")
        print(f"Saved chunks to {chunks_path}")
        print(f"Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str, chunks_path: str, metadata_path: str):
        """
        Load a previously saved FAISS index and data.
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks file
            metadata_path: Path to metadata file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunks")


def download_arxiv_papers(category: str = 'cs.CL', max_results: int = 50, 
                          output_dir: str = 'arxiv_papers') -> List[str]:
    """
    Download arXiv papers using the arXiv API.
    
    Args:
        category: arXiv category (e.g., 'cs.CL')
        max_results: Number of papers to download
        output_dir: Directory to save PDFs
        
    Returns:
        List of paths to downloaded PDFs
    """
    import arxiv
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Downloading {max_results} papers from {category}...")
    
    # Search for papers
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    pdf_paths = []
    
    for paper in tqdm(client.results(search), total=max_results, desc="Downloading"):
        try:
            # Download PDF
            pdf_path = output_path / f"{paper.get_short_id()}.pdf"
            paper.download_pdf(filename=str(pdf_path))
            pdf_paths.append(str(pdf_path))
            time.sleep(1)  # Be nice to arXiv servers
        except Exception as e:
            print(f"Error downloading {paper.title}: {e}")
            continue
    
    print(f"Downloaded {len(pdf_paths)} papers to {output_dir}")
    return pdf_paths


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RAG Pipeline for arXiv Papers")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ArXivRAGPipeline(model_name='all-MiniLM-L6-v2')
    
    print("\nPipeline implementation complete!")
