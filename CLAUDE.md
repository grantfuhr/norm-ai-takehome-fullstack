# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a fullstack take-home assignment implementing a Game of Thrones laws querying system:

**Backend (FastAPI)**: A FastAPI service that processes queries about fictional Game of Thrones laws using vector search and LLM responses. The system uses:
- QdrantVectorStore for semantic search over law documents
- OpenAI embeddings and GPT-4 for query processing 
- CitationQueryEngine pattern to provide sources with responses
- PDF document processing from `docs/laws.pdf`

## Development Commands

### Backend
```bash
# Run tests 
uv run pytest ...

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Build and run Docker container
docker build -t norm-ai-takehome .
docker run -e OPENAI_API_KEY=your_key -p 8000:80 norm-ai-takehome
```


## Environment Setup

Backend requires `OPENAI_API_KEY` environment variable for OpenAI services.

## Implementation Notes

The backend service skeleton exists but needs completion of:
- DocumentService.create_documents() method to extract PDF content
- QdrantService.query() method to implement citation-based querying
- FastAPI endpoint creation to accept query strings and return JSON responses

Frontend provides basic Next.js structure ready for client implementation.
