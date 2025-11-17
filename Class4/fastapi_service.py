from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from rag_pipeline import ArXivRAGPipeline
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="arXiv RAG Search API",
    description="Retrieval-Augmented Generation API for searching arXiv papers",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None


class SearchResult(BaseModel):
    """Model for search result"""
    rank: int
    chunk: str
    distance: float
    paper: str
    chunk_id: int


class SearchResponse(BaseModel):
    """Model for search response"""
    query: str
    num_results: int
    results: List[SearchResult]


@app.on_event("startup")
async def startup_event():
    """Load the RAG pipeline on startup"""
    global pipeline
    
    print("Loading RAG pipeline...")
    pipeline = ArXivRAGPipeline(model_name='all-MiniLM-L6-v2')
    
    # Try to load pre-built index
    try:
        pipeline.load_index(
            'faiss_index.bin',
            'chunks.json',
            'metadata.json'
        )
        print("Successfully loaded pre-built index")
    except Exception as e:
        print(f"Warning: Could not load pre-built index: {e}")
        print("You'll need to build the index first by running rag_pipeline.py")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "arXiv RAG Search API",
        "endpoints": {
            "/search": "Search for relevant passages (GET)",
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
    
    return {
        "status": "healthy",
        "index_size": pipeline.index.ntotal,
        "num_chunks": len(pipeline.chunks)
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed data"""
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Count papers
    unique_papers = set(meta['paper'] for meta in pipeline.metadata)
    
    return {
        "total_chunks": len(pipeline.chunks),
        "total_papers": len(unique_papers),
        "index_size": pipeline.index.ntotal,
        "embedding_dimension": pipeline.embedding_dim,
        "papers": sorted(list(unique_papers))
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Search for relevant passages given a query.
    
    Args:
        q: The search query
        k: Number of top results to return (default: 3, max: 20)
        
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
        results = pipeline.search(q, k=k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                rank=result['rank'],
                chunk=result['chunk'],
                distance=result['distance'],
                paper=result['metadata']['paper'],
                chunk_id=result['metadata']['chunk_id']
            ))
        
        return SearchResponse(
            query=q,
            num_results=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/paper")
async def search_by_paper(
    paper_name: str = Query(..., description="Paper name to filter by"),
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results", ge=1, le=20)
):
    """
    Search within a specific paper.
    
    Args:
        paper_name: Name of the paper to search within
        q: The search query
        k: Number of results to return
        
    Returns:
        Search results filtered to the specified paper
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        # Get more results than needed for filtering
        all_results = pipeline.search(q, k=k*5)
        
        # Filter by paper name
        filtered_results = [
            r for r in all_results 
            if r['metadata']['paper'] == paper_name
        ][:k]
        
        if not filtered_results:
            return {
                "query": q,
                "paper": paper_name,
                "num_results": 0,
                "results": [],
                "message": f"No results found in paper '{paper_name}'"
            }
        
        return {
            "query": q,
            "paper": paper_name,
            "num_results": len(filtered_results),
            "results": [
                {
                    "rank": i + 1,
                    "chunk": r['chunk'],
                    "distance": r['distance'],
                    "chunk_id": r['metadata']['chunk_id']
                }
                for i, r in enumerate(filtered_results)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    print("Starting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
