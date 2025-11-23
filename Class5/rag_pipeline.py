import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json
import pickle
from pathlib import Path
import requests
import time
from tqdm import tqdm
import sqlite3
from datetime import datetime


class ArXivRAGPipeline:
    """Complete RAG pipeline for arXiv papers with hybrid retrieval (FAISS + SQLite FTS5)"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', db_path: str = 'arxiv_hybrid.db'):
        """
        Initialize the RAG pipeline with an embedding model and SQLite database.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            db_path: Path to the SQLite database file
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunks = []
        self.metadata = []
        self.index = None
        self.db_path = db_path
        self.conn = None
        
    def _init_database(self):
        """Initialize SQLite database with document and FTS5 tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create documents table for metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                author TEXT,
                year INTEGER,
                keywords TEXT,
                source_path TEXT,
                chunk_id INTEGER,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                content,
                content='documents',
                content_rowid='doc_id'
            )
        """)
        
        # Create triggers to keep FTS5 index in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO doc_chunks(rowid, content) VALUES (new.doc_id, new.content);
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO doc_chunks(doc_chunks, rowid, content) VALUES('delete', old.doc_id, old.content);
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO doc_chunks(doc_chunks, rowid, content) VALUES('delete', old.doc_id, old.content);
                INSERT INTO doc_chunks(rowid, content) VALUES (new.doc_id, new.content);
            END;
        """)
        
        self.conn.commit()
        print(f"Database initialized at {self.db_path}")
        
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
            page_text = page.get_text()
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
            if len(chunk_text.strip()) > 50:
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
        
        # Using IndexFlatL2 for exact search
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        print(f"Index built with {index.ntotal} vectors")
        return index
    
    def process_papers(self, pdf_paths: List[str], chunk_size: int = 512, overlap: int = 50):
        """
        Process multiple papers: extract text, chunk, embed, index in FAISS and SQLite.
        
        Args:
            pdf_paths: List of paths to PDF files
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        # Initialize database
        self._init_database()
        
        all_chunks = []
        all_metadata = []
        
        print(f"\nProcessing {len(pdf_paths)} papers...")
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            try:
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                # Chunk text
                chunks = self.chunk_text(text, max_tokens=chunk_size, overlap=overlap)
                
                # Extract metadata (you can enhance this with actual metadata extraction)
                paper_name = Path(pdf_path).stem
                author = "Unknown"  # Could extract from PDF metadata
                year = datetime.now().year  # Could extract from PDF metadata
                keywords = paper_name  # Could use keyword extraction
                
                # Store in SQLite and prepare for FAISS
                cursor = self.conn.cursor()
                for chunk_idx, chunk in enumerate(chunks):
                    # Insert into SQLite
                    cursor.execute("""
                        INSERT INTO documents (title, author, year, keywords, source_path, chunk_id, content)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (paper_name, author, year, keywords, pdf_path, chunk_idx, chunk))
                    
                    doc_id = cursor.lastrowid
                    
                    # Store for FAISS indexing
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'doc_id': doc_id,
                        'paper': paper_name,
                        'chunk_id': chunk_idx,
                        'source_path': pdf_path,
                        'author': author,
                        'year': year,
                        'keywords': keywords
                    })
                
                self.conn.commit()
                    
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
    
    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform keyword search using SQLite FTS5.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results with BM25 scores
        """
        if self.conn is None:
            raise ValueError("Database not initialized. Call process_papers() first.")
        
        cursor = self.conn.cursor()
        
        # Clean and prepare the FTS5 query
        # FTS5 needs quotes around phrases to treat them as search terms, not column names
        fts_query = query.replace('"', '""')  # Escape existing quotes
        
        try:
            # Try as a phrase query first (most reliable for multi-word queries)
            cursor.execute("""
                SELECT 
                    d.doc_id,
                    d.content,
                    d.title,
                    d.author,
                    d.year,
                    d.chunk_id,
                    d.source_path,
                    bm25(doc_chunks) as score
                FROM documents d
                JOIN doc_chunks ON d.doc_id = doc_chunks.rowid
                WHERE doc_chunks MATCH ?
                ORDER BY score
                LIMIT ?
            """, (f'"{fts_query}"', k))
            
            results = cursor.fetchall()
            
            # If no results with phrase query, try OR query with individual terms
            if not results:
                terms = fts_query.split()
                or_query = ' OR '.join([f'"{term}"' for term in terms if term.strip()])
                cursor.execute("""
                    SELECT 
                        d.doc_id,
                        d.content,
                        d.title,
                        d.author,
                        d.year,
                        d.chunk_id,
                        d.source_path,
                        bm25(doc_chunks) as score
                    FROM documents d
                    JOIN doc_chunks ON d.doc_id = doc_chunks.rowid
                    WHERE doc_chunks MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (or_query, k))
                results = cursor.fetchall()
        
        except Exception as e:
            print(f"Warning: FTS5 query failed for '{query}': {e}")
            print(f"Falling back to LIKE query...")
            
            # Fallback to LIKE query if FTS5 fails completely
            cursor.execute("""
                SELECT 
                    d.doc_id,
                    d.content,
                    d.title,
                    d.author,
                    d.year,
                    d.chunk_id,
                    d.source_path,
                    0.0 as score
                FROM documents d
                WHERE d.content LIKE ?
                LIMIT ?
            """, (f'%{query}%', k))
            results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'doc_id': row[0],
                'chunk': row[1],
                'paper': row[2],
                'author': row[3],
                'year': row[4],
                'chunk_id': row[5],
                'source_path': row[6],
                'score': row[7]
            })
        
        return formatted_results

    
    def vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform vector search using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results with distances
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call process_papers() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    'idx': int(idx),
                    'chunk': self.chunks[idx],
                    'distance': float(dist),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Array of scores
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Normalized scores
        """
        if len(scores) == 0:
            return scores
        
        if method == 'minmax':
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score - min_score == 0:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)
        elif method == 'zscore':
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return np.ones_like(scores)
            return (scores - mean) / std
        else:
            return scores
    
    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_list: List of ranked result lists
            k: RRF constant (default 60)
            
        Returns:
            Merged and re-ranked results
        """
        # Track RRF scores by document identifier
        rrf_scores = {}
        doc_data = {}
        
        for results in results_list:
            for rank, result in enumerate(results, start=1):
                # Use doc_id if available, otherwise use chunk text as identifier
                doc_id = result.get('doc_id', result.get('chunk', ''))
                # Convert doc_id to string if it's an integer, then truncate for key
                doc_key = str(doc_id)[:100] if isinstance(doc_id, int) else doc_id[:100]
                
                # Calculate RRF score
                rrf_score = 1.0 / (k + rank)
                
                if doc_key in rrf_scores:
                    rrf_scores[doc_key] += rrf_score
                else:
                    rrf_scores[doc_key] = rrf_score
                    doc_data[doc_key] = result
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare final results
        merged_results = []
        for doc_key, score in sorted_docs:
            result = doc_data[doc_key].copy()
            result['rrf_score'] = score
            merged_results.append(result)
        
        return merged_results
    
    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.5, 
                     method: str = 'weighted') -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            k: Number of final results to return
            alpha: Weight for vector search (1-alpha for keyword). Only used for 'weighted' method.
            method: Fusion method ('weighted' or 'rrf')
            
        Returns:
            List of top-k hybrid results
        """
        # Get results from both methods (retrieve more for better fusion)
        search_k = max(k * 3, 10)
        
        vector_results = self.vector_search(query, k=search_k)
        keyword_results = self.keyword_search(query, k=search_k)
        
        if method == 'rrf':
            # Use Reciprocal Rank Fusion
            merged = self.reciprocal_rank_fusion([vector_results, keyword_results])
            return merged[:k]
        
        elif method == 'weighted':
            # Weighted score fusion
            # Normalize vector distances (lower is better, so we invert)
            vector_scores = np.array([r['distance'] for r in vector_results])
            # For distances, lower is better, so we invert after normalization
            if len(vector_scores) > 0:
                vector_scores_norm = 1 - self.normalize_scores(vector_scores, method='minmax')
            else:
                vector_scores_norm = np.array([])
            
            # Normalize keyword scores (higher is better for BM25)
            keyword_scores = np.array([r['score'] for r in keyword_results])
            if len(keyword_scores) > 0:
                # BM25 scores are negative, so we invert them
                keyword_scores_norm = self.normalize_scores(-keyword_scores, method='minmax')
            else:
                keyword_scores_norm = np.array([])
            
            # Combine results with weighted scores
            combined = {}
            
            # Add vector results
            for i, result in enumerate(vector_results):
                chunk_key = result['chunk'][:100]
                if i < len(vector_scores_norm):
                    combined[chunk_key] = {
                        'result': result,
                        'vector_score': vector_scores_norm[i],
                        'keyword_score': 0.0
                    }
            
            # Add/update keyword results
            for i, result in enumerate(keyword_results):
                chunk_key = result['chunk'][:100]
                if i < len(keyword_scores_norm):
                    if chunk_key in combined:
                        combined[chunk_key]['keyword_score'] = keyword_scores_norm[i]
                    else:
                        combined[chunk_key] = {
                            'result': result,
                            'vector_score': 0.0,
                            'keyword_score': keyword_scores_norm[i]
                        }
            
            # Calculate hybrid scores
            for key in combined:
                v_score = combined[key]['vector_score']
                k_score = combined[key]['keyword_score']
                combined[key]['hybrid_score'] = alpha * v_score + (1 - alpha) * k_score
            
            # Sort by hybrid score
            sorted_results = sorted(combined.items(), 
                                  key=lambda x: x[1]['hybrid_score'], 
                                  reverse=True)
            
            # Prepare final results
            final_results = []
            for chunk_key, data in sorted_results[:k]:
                result = data['result'].copy()
                result['hybrid_score'] = data['hybrid_score']
                result['vector_score'] = data['vector_score']
                result['keyword_score'] = data['keyword_score']
                final_results.append(result)
            
            return final_results
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'weighted' or 'rrf'.")
    
    def search(self, query: str, k: int = 3, method: str = 'hybrid', **kwargs) -> List[Dict]:
        """
        Unified search interface supporting vector, keyword, and hybrid search.
        
        Args:
            query: Search query
            k: Number of results to return
            method: Search method ('vector', 'keyword', or 'hybrid')
            **kwargs: Additional arguments for hybrid search (alpha, fusion_method)
            
        Returns:
            List of search results
        """
        if method == 'vector':
            results = self.vector_search(query, k=k)
            # Add rank
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results
        
        elif method == 'keyword':
            results = self.keyword_search(query, k=k)
            # Add rank
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results
        
        elif method == 'hybrid':
            alpha = kwargs.get('alpha', 0.5)
            fusion_method = kwargs.get('fusion_method', 'weighted')
            results = self.hybrid_search(query, k=k, alpha=alpha, method=fusion_method)
            # Add rank
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'vector', 'keyword', or 'hybrid'.")
    
    def evaluate(self, test_queries: List[Dict[str, any]], k: int = 3) -> Dict[str, float]:
        """
        Evaluate retrieval performance using recall/hit rate metrics.
        
        Args:
            test_queries: List of dicts with 'query' and 'relevant_docs' (list of paper names)
            k: Number of top results to consider
            
        Returns:
            Dictionary with evaluation metrics for each method
        """
        methods = ['vector', 'keyword', 'hybrid']
        results = {method: {'hits': 0, 'total': 0} for method in methods}
        
        print(f"\nEvaluating on {len(test_queries)} queries...")
        
        for query_data in tqdm(test_queries, desc="Evaluating"):
            query = query_data['query']
            relevant_docs = set(query_data['relevant_docs'])
            
            for method in methods:
                search_results = self.search(query, k=k, method=method)
                
                # Check if any relevant document is in top-k
                retrieved_docs = set()
                for result in search_results:
                    if 'paper' in result:
                        retrieved_docs.add(result['paper'])
                    elif 'metadata' in result and 'paper' in result['metadata']:
                        retrieved_docs.add(result['metadata']['paper'])
                
                # Check for hit
                if len(relevant_docs & retrieved_docs) > 0:
                    results[method]['hits'] += 1
                
                results[method]['total'] += 1
        
        # Calculate hit rates
        metrics = {}
        for method in methods:
            hit_rate = results[method]['hits'] / results[method]['total'] if results[method]['total'] > 0 else 0
            metrics[f'{method}_recall@{k}'] = hit_rate
            print(f"{method.capitalize()} Recall@{k}: {hit_rate:.3f}")
        
        return metrics
    
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
        print(f"SQLite database at {self.db_path}")
    
    def load_index(self, index_path: str, chunks_path: str, metadata_path: str, db_path: Optional[str] = None):
        """
        Load a previously saved FAISS index and data.
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks file
            metadata_path: Path to metadata file
            db_path: Optional path to SQLite database
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Connect to database
        if db_path:
            self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunks")
        print(f"Connected to database at {self.db_path}")


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
    print("Hybrid RAG Pipeline for arXiv Papers")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ArXivRAGPipeline(model_name='all-MiniLM-L6-v2', db_path='arxiv_hybrid.db')
    
    print("\nHybrid pipeline implementation complete!")
    print("\nFeatures:")
    print("- FAISS vector search")
    print("- SQLite FTS5 keyword search")
    print("- Hybrid retrieval (weighted & RRF)")
    print("- Evaluation metrics (Recall@k)")