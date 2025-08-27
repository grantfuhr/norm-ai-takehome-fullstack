from app.document import DocumentService
from app.qdrant import QdrantService


if __name__ == "__main__":
    # Example workflow
    doc_service = DocumentService() # implemented
    docs = doc_service.create_documents() # Now implemented

    index = QdrantService() # implemented
    index.connect() # implemented
    index.load(docs) # implemented

    out = index.query("what happens if I steal?")
    print(out)





