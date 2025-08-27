import os

import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine
from app.datatypes import Output, Citation, Input

key = os.environ['OPENAI_API_KEY']

class QdrantService:
    def __init__(self, k: int = 5):
        self.index = None
        self.k = k
    
    def connect(self) -> None:
        # TODO should I use the docker version and a docker-compose setup?
        client = qdrant_client.QdrantClient(location=":memory:")
                
        vstore = QdrantVectorStore(client=client, collection_name='temp')

        # Configure settings instead of ServiceContext
        from llama_index.core import Settings
        Settings.embed_model = OpenAIEmbedding()
        Settings.llm = OpenAI(api_key=key, model="gpt-4")

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vstore
            )

    def load(self, docs = list[Document]):
        self.index.insert_nodes(docs)
    
    def query(self, query_str: str) -> Output:
        # Initialize CitationQueryEngine with similarity_top_k parameter
        query_engine = CitationQueryEngine.from_args(
            index=self.index,
            similarity_top_k=self.k
        )
        
        # Run the query
        response = query_engine.query(query_str)
        
        # Extract citations from the response
        citations = []
        if hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes, 1):
                # Create a more informative source using the metadata
                full_path = node.metadata.get('full_path', 'Unknown section')
                page_num = node.metadata.get('page_number', 'Unknown page')
                source_info = f"{full_path} (Page {page_num})"
                
                citation = Citation(
                    source=source_info,
                    text=f"Source {i}:\n{node.text}"
                )
                citations.append(citation)
        
        # Create and return Output object
        output = Output(
            query=query_str,
            response=str(response),
            citations=citations
        )
        
        return output
       

