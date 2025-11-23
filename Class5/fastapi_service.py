from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from rag_pipeline import ArXivRAGPipeline
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="arXiv Hybrid RAG Search API",
    description="Retrieval-Augmented Generation API with hybrid search (FAISS + SQLite FTS5)",
    version="2.0.0"
)

# Global pipeline instance
pipeline = None


class SearchResult(BaseModel):
    """Model for search result"""
    rank: int
    chunk: str
    paper: str
    score: Optional[float] = None
    distance: Optional[float] = None
    hybrid_score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Model for search response"""
    query: str
    method: str
    num_results: int
    results: List[SearchResult]


class EvaluationRequest(BaseModel):
    """Model for evaluation request"""
    queries: List[dict]  # List of {'query': str, 'relevant_docs': List[str]}
    k: int = 3


class EvaluationResponse(BaseModel):
    """Model for evaluation response"""
    metrics: dict
    num_queries: int


@app.on_event("startup")
async def startup_event():
    """Load the RAG pipeline on startup"""
    global pipeline
    
    print("Loading Hybrid RAG pipeline...")
    pipeline = ArXivRAGPipeline(model_name='all-MiniLM-L6-v2', db_path='arxiv_hybrid.db')
    
    # Try to load pre-built index
    try:
        pipeline.load_index(
            'faiss_index.bin',
            'chunks.json',
            'metadata.json',
            db_path='arxiv_hybrid.db'
        )
        print("Successfully loaded pre-built index")
    except Exception as e:
        print(f"Warning: Could not load pre-built index: {e}")
        print("You'll need to build the index first by running the pipeline")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "arXiv Hybrid RAG Search API",
        "version": "2.0.0",
        "features": [
            "FAISS vector search",
            "SQLite FTS5 keyword search",
            "Hybrid retrieval (weighted + RRF)",
            "Evaluation metrics"
        ],
        "endpoints": {
            "/search": "Unified search endpoint (GET)",
            "/hybrid_search": "Hybrid search with customization (GET)",
            "/vector_search": "Pure vector search (GET)",
            "/keyword_search": "Pure keyword search (GET)",
            "/evaluate": "Evaluate retrieval performance (POST)",
            "/health": "Check API health (GET)",
            "/stats": "Get index statistics (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None or pipeline.index is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Index not loaded"}
        )
    
    # Check database connection
    db_ok = pipeline.conn is not None
    
    return {
        "status": "healthy",
        "index_size": pipeline.index.ntotal,
        "num_chunks": len(pipeline.chunks),
        "database_connected": db_ok
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed data"""
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Count papers
    unique_papers = set(meta['paper'] for meta in pipeline.metadata)
    
    # Get database stats
    cursor = pipeline.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    db_chunks = cursor.fetchone()[0]
    
    return {
        "total_chunks": len(pipeline.chunks),
        "db_chunks": db_chunks,
        "total_papers": len(unique_papers),
        "index_size": pipeline.index.ntotal,
        "embedding_dimension": pipeline.embedding_dim,
        "database_path": pipeline.db_path,
        "papers": sorted(list(unique_papers))
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20),
    method: str = Query("hybrid", description="Search method: 'vector', 'keyword', or 'hybrid'"),
    alpha: float = Query(0.5, description="Weight for vector search (hybrid only)", ge=0.0, le=1.0),
    fusion: str = Query("weighted", description="Fusion method: 'weighted' or 'rrf' (hybrid only)")
):
    """
    Unified search endpoint supporting vector, keyword, and hybrid search.
    
    Args:
        q: The search query
        k: Number of top results to return (default: 3, max: 20)
        method: Search method ('vector', 'keyword', or 'hybrid')
        alpha: Weight for vector search in hybrid mode (0-1)
        fusion: Fusion method for hybrid search ('weighted' or 'rrf')
        
    Returns:
        SearchResponse containing the query and top-k results
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Please build the index first."
        )
    
    try:
        # Perform search
        results = pipeline.search(
            q, 
            k=k, 
            method=method,
            alpha=alpha,
            fusion_method=fusion
        )
        
        # Format results
        formatted_results = []
        for result in results:
            search_result = SearchResult(
                rank=result.get('rank', 0),
                chunk=result.get('chunk', ''),
                paper=result.get('paper', result.get('metadata', {}).get('paper', 'Unknown')),
                distance=result.get('distance'),
                score=result.get('score'),
                hybrid_score=result.get('hybrid_score'),
                vector_score=result.get('vector_score'),
                keyword_score=result.get('keyword_score')
            )
            formatted_results.append(search_result)
        
        return SearchResponse(
            query=q,
            method=method,
            num_results=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20),
    alpha: float = Query(0.5, description="Weight for vector search (0-1)", ge=0.0, le=1.0),
    method: str = Query("weighted", description="Fusion method: 'weighted' or 'rrf'")
):
    """
    Hybrid search endpoint combining vector and keyword search.
    
    Args:
        q: The search query
        k: Number of top results to return
        alpha: Weight for vector search (1-alpha for keyword)
        method: Fusion method ('weighted' or 'rrf')
        
    Returns:
        SearchResponse with hybrid results
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        results = pipeline.hybrid_search(q, k=k, alpha=alpha, method=method)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(SearchResult(
                rank=i,
                chunk=result.get('chunk', ''),
                paper=result.get('paper', result.get('metadata', {}).get('paper', 'Unknown')),
                hybrid_score=result.get('hybrid_score') or result.get('rrf_score'),
                vector_score=result.get('vector_score'),
                keyword_score=result.get('keyword_score'),
                distance=result.get('distance'),
                score=result.get('score')
            ))
        
        return SearchResponse(
            query=q,
            method=f"hybrid_{method}",
            num_results=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.get("/vector_search", response_model=SearchResponse)
async def vector_search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results", ge=1, le=20)
):
    """Pure vector search using FAISS."""
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        results = pipeline.vector_search(q, k=k)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(SearchResult(
                rank=i,
                chunk=result['chunk'],
                paper=result['metadata']['paper'],
                distance=result['distance']
            ))
        
        return SearchResponse(
            query=q,
            method="vector",
            num_results=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@app.get("/keyword_search", response_model=SearchResponse)
async def keyword_search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results", ge=1, le=20)
):
    """Pure keyword search using SQLite FTS5."""
    if pipeline is None or pipeline.conn is None:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    try:
        results = pipeline.keyword_search(q, k=k)
        
        if not results:
            return SearchResponse(
                query=q,
                method="keyword",
                num_results=0,
                results=[]
            )
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(SearchResult(
                rank=i,
                chunk=result['chunk'],
                paper=result['paper'],
                score=result['score']
            ))
        
        return SearchResponse(
            query=q,
            method="keyword",
            num_results=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate retrieval performance on test queries.
    
    Args:
        request: EvaluationRequest with queries and k value
        
    Returns:
        EvaluationResponse with metrics
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        metrics = pipeline.evaluate(request.queries, k=request.k)
        
        return EvaluationResponse(
            metrics=metrics,
            num_queries=len(request.queries)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/search/paper")
async def search_by_paper(
    paper_name: str = Query(..., description="Paper name to filter by"),
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results", ge=1, le=20),
    method: str = Query("hybrid", description="Search method")
):
    """
    Search within a specific paper.
    
    Args:
        paper_name: Name of the paper to search within
        q: The search query
        k: Number of results to return
        method: Search method to use
        
    Returns:
        Search results filtered to the specified paper
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        # Get more results than needed for filtering
        all_results = pipeline.search(q, k=k*5, method=method)
        
        # Filter by paper name
        filtered_results = []
        for r in all_results:
            paper = r.get('paper', r.get('metadata', {}).get('paper', ''))
            if paper == paper_name:
                filtered_results.append(r)
                if len(filtered_results) >= k:
                    break
        
        if not filtered_results:
            return {
                "query": q,
                "paper": paper_name,
                "method": method,
                "num_results": 0,
                "results": [],
                "message": f"No results found in paper '{paper_name}'"
            }
        
        return {
            "query": q,
            "paper": paper_name,
            "method": method,
            "num_results": len(filtered_results),
            "results": [
                {
                    "rank": i + 1,
                    "chunk": r.get('chunk', ''),
                    "distance": r.get('distance'),
                    "score": r.get('score'),
                    "hybrid_score": r.get('hybrid_score')
                }
                for i, r in enumerate(filtered_results)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    print("Starting Hybrid RAG FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
