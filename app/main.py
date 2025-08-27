from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from app.document import DocumentService
from app.qdrant import QdrantService
from app.datatypes import Output

# Initialize services
doc_service = DocumentService()
qdrant_service = QdrantService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and load documents into the vector store on startup"""
    try:
        # Process PDF documents
        docs = doc_service.create_documents()
        
        # Initialize Qdrant and load documents
        qdrant_service.connect()
        qdrant_service.load(docs)
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize services: {str(e)}")
    
    yield
    # Cleanup code would go here if needed

app = FastAPI(lifespan=lifespan)

@app.get("/query", response_model=Output)
async def query_laws(q: str = Query(..., description="Query string to search the laws")):
    """
    Query endpoint that accepts a query string and returns a JSON response 
    with the answer and citations from the Game of Thrones laws.
    """
    try:
        # Process the query and return results
        result = qdrant_service.query(q)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)