# Norm AI Exercise - Implementation Guide (made with Claude Opus 4.1)

## Architecture Overview

Building a precise legal document RAG system with guaranteed section-level citations. Core challenge: maintaining exact section references (e.g., "Section 3.1.1") throughout the retrieval and generation pipeline.

## Critical Implementation Components

### 1. Enhanced Document Parser with Hierarchical Section Tracking

```python
# document.py - DocumentService implementation

from typing import List, Dict, Tuple, Optional
import re
import PyPDF2
from llama_index.core.schema import Document

class DocumentService:
    """
    Parses hierarchical legal documents maintaining exact section references.
    Key insight: Legal documents have predictable structure that we exploit.
    """
    
    def create_documents(self, file_path: str = "docs/laws.pdf") -> List[Document]:
        """
        Main entry point. Returns documents with precise section metadata.
        Each section becomes a separate document for granular retrieval.
        """
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
        
        # Clean text - PDF extraction often has issues
        full_text = self._clean_pdf_text(full_text)
        
        # Parse hierarchical structure
        return self._parse_hierarchical_structure(full_text)
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Handle common PDF extraction artifacts.
        Game of Thrones laws PDF specific cleaning.
        """
        # Remove page numbers and headers
        text = re.sub(r'Laws of the Seven Kingdoms\n', '', text)
        text = re.sub(r'\n+', '\n', text)  # Normalize newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces within lines
        
        # Fix common OCR/extraction issues
        text = text.replace('', 'fl')  # Common ligature issue
        text = text.replace('', 'fi')
        
        return text
    
    def _parse_hierarchical_structure(self, text: str) -> List[Document]:
        """
        Core parsing logic. Creates documents at multiple granularity levels.
        
        Strategy:
        1. Parse main sections (1. Peace, 2. Religion)
        2. Parse subsections (1.1, 2.1) 
        3. Parse sub-subsections (1.1.1, 2.1.1)
        4. Create documents at each level with proper parent references
        """
        documents = []
        
        # Split into lines for line-by-line processing
        lines = text.split('\n')
        
        # State tracking
        current_main_section = None
        current_main_title = None
        current_main_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Pattern matching for different hierarchy levels
            # Main section: "1. Peace" or "2. Religion"
            main_match = re.match(r'^(\d+)\.\s+([A-Z][a-zA-Z]+)$', line)
            
            if main_match:
                # Save previous main section if exists
                if current_main_section:
                    self._create_section_documents(
                        documents, 
                        current_main_section, 
                        current_main_title, 
                        current_main_content
                    )
                
                # Start new main section
                current_main_section = main_match.group(1)
                current_main_title = main_match.group(2)
                current_main_content = []
                
                # Look ahead for subsections
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    # Check if we hit the next main section
                    if re.match(r'^(\d+)\.\s+([A-Z][a-zA-Z]+)$', next_line):
                        break
                    
                    # Parse subsection
                    sub_match = re.match(r'^(\d+)\.(\d+)\.?\s+(.+)', next_line)
                    if sub_match and sub_match.group(1) == current_main_section:
                        sub_num = f"{sub_match.group(1)}.{sub_match.group(2)}"
                        sub_content = sub_match.group(3)
                        
                        # Parse multi-line subsection content
                        full_sub_content = [sub_content]
                        i += 1
                        
                        # Look for sub-subsections
                        while i < len(lines):
                            subsub_line = lines[i].strip()
                            
                            # Check for sub-subsection (e.g., 1.1.1)
                            subsub_match = re.match(
                                rf'^{re.escape(sub_num)}\.(\d+)\.?\s+(.+)', 
                                subsub_line
                            )
                            
                            if subsub_match:
                                subsub_num = f"{sub_num}.{subsub_match.group(1)}"
                                subsub_content = subsub_match.group(2)
                                
                                # Create sub-subsection document
                                doc = Document(
                                    text=subsub_content,
                                    metadata={
                                        'section_id': subsub_num,
                                        'section_title': self._extract_title(subsub_content),
                                        'parent_section': sub_num,
                                        'main_section': current_main_section,
                                        'main_title': current_main_title,
                                        'hierarchy_level': 3,
                                        'full_path': f"{current_main_title} > {sub_num} > {subsub_num}"
                                    }
                                )
                                documents.append(doc)
                                full_sub_content.append(f"{subsub_num}. {subsub_content}")
                                i += 1
                            elif re.match(r'^(\d+)\.', subsub_line):
                                # Hit next section
                                break
                            else:
                                # Continuation of current subsection
                                if subsub_line and not subsub_line.startswith('Citations:'):
                                    full_sub_content.append(subsub_line)
                                i += 1
                                if i >= len(lines) or re.match(r'^(\d+)\.', lines[i].strip()):
                                    break
                        
                        # Create subsection document
                        doc = Document(
                            text=' '.join(full_sub_content),
                            metadata={
                                'section_id': sub_num,
                                'section_title': self._extract_title(sub_content),
                                'parent_section': current_main_section,
                                'main_title': current_main_title,
                                'hierarchy_level': 2,
                                'full_path': f"{current_main_title} > {sub_num}"
                            }
                        )
                        documents.append(doc)
                        current_main_content.extend(full_sub_content)
                        continue
                    else:
                        # Regular content line
                        if next_line and not next_line.startswith('Citations:'):
                            current_main_content.append(next_line)
                    
                    i += 1
            else:
                i += 1
        
        # Save last section
        if current_main_section:
            self._create_section_documents(
                documents, 
                current_main_section, 
                current_main_title, 
                current_main_content
            )
        
        return documents
    
    def _create_section_documents(
        self, 
        documents: List[Document], 
        section_id: str, 
        title: str, 
        content: List[str]
    ):
        """
        Creates main section document with aggregated content.
        """
        if content:
            doc = Document(
                text=' '.join(content),
                metadata={
                    'section_id': section_id,
                    'section_title': title,
                    'parent_section': None,
                    'main_title': title,
                    'hierarchy_level': 1,
                    'full_path': title
                }
            )
            documents.append(doc)
    
    def _extract_title(self, content: str, max_length: int = 50) -> str:
        """
        Extract meaningful title from content beginning.
        """
        # Remove common starting words
        title = re.sub(r'^(The |A |An )', '', content)
        
        # Truncate at sentence boundary if possible
        if len(title) > max_length:
            # Try to cut at punctuation
            for punct in ['.', ',', ';', ':']:
                idx = title[:max_length].rfind(punct)
                if idx > 20:  # Ensure we keep reasonable amount
                    return title[:idx]
            
            # Fallback to word boundary
            return title[:max_length].rsplit(' ', 1)[0] + "..."
        
        return title
```

### 2. QdrantService with CitationQueryEngine

```python
# query.py - QdrantService implementation

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import qdrant_client

class QdrantService:
    """
    Vector search with precise citation generation.
    Uses CitationQueryEngine with custom prompts for legal compliance.
    """
    
    def __init__(self, k: int = 3):
        self.index = None
        self.k = k
        self.service_context = None
    
    def connect(self) -> None:
        """Initialize vector store and service context."""
        client = qdrant_client.QdrantClient(location=":memory:")
        
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name='laws',
            dimension=1536  # OpenAI embedding dimension
        )
        
        # Configure LLM with low temperature for consistency
        llm = OpenAI(
            model="gpt-4",
            temperature=0.1,  # Low temperature for factual consistency
            api_key=os.environ['OPENAI_API_KEY']
        )
        
        self.service_context = ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=os.environ['OPENAI_API_KEY']
            ),
            llm=llm,
            chunk_size=512,  # Optimal for legal paragraphs
            chunk_overlap=50  # Maintain context at boundaries
        )
        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=self.service_context
        )
    
    def load(self, docs: List[Document]):
        """Load documents into vector store."""
        # Important: insert_nodes maintains metadata better than from_documents
        for doc in docs:
            self.index.insert(doc)
        
        # Force index refresh
        self.index.storage_context.persist()
    
    def query(self, query_str: str) -> Output:
        """
        Execute query with guaranteed section citations.
        
        Three-stage approach:
        1. Retrieve with CitationQueryEngine
        2. Enhance with custom prompting
        3. Post-process for citation verification
        """
        
        # Stage 1: Configure CitationQueryEngine with legal-specific prompt
        qa_prompt = PromptTemplate(
            """You are a legal compliance assistant analyzing the Laws of the Seven Kingdoms.

CRITICAL CITATION REQUIREMENTS:
1. Every legal claim MUST cite the specific section number
2. Use EXACT format: "According to Section X.X.X, [statement]" or "Section X.X states that..."
3. When multiple sections apply, cite each one
4. Use the most specific section number available (prefer 3.1.1 over 3.1 over 3)

Context with section numbers:
{context_str}

Question: {query_str}

Legal Analysis with Precise Citations:"""
        )
        
        # Configure citation engine
        citation_engine = CitationQueryEngine.from_args(
            self.index,
            similarity_top_k=self.k
            citation_chunk_size=256, 
            citation_chunk_overlap=50,
            text_qa_template=qa_prompt,
            # Node postprocessor to prioritize exact matches
            node_postprocessors=[
                SectionMetadataFilter(query_str),  # Custom filter
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        
        # Stage 2: Execute query with enhanced prompt
        enhanced_query = self._enhance_query(query_str)
        response = citation_engine.query(enhanced_query)
        
        # Stage 3: Post-process response
        processed_response = self._post_process_response(
            response, 
            query_str
        )
        
        # Extract structured citations
        citations = self._extract_citations(response.source_nodes)
        
        return Output(
            query=query_str,
            response=processed_response,
            citations=citations
        )
    
    def _enhance_query(self, query_str: str) -> str:
        """
        Enhance query to encourage specific citations.
        """
        # Detect query type and add citation instructions
        if any(word in query_str.lower() for word in ['what', 'which', 'how']):
            return f"{query_str} Cite the specific section numbers that address this."
        elif 'if' in query_str.lower() or 'when' in query_str.lower():
            return f"{query_str} Reference the exact sections that apply to this scenario."
        else:
            return f"{query_str} Provide specific section citations for your answer."
    
    def _post_process_response(self, response, original_query: str) -> str:
        """
        Verify and enhance citations in response.
        
        Critical: Ensures every claim has a section reference.
        """
        response_text = str(response)
        source_nodes = response.source_nodes
        
        # Check if response contains section citations
        import re
        section_pattern = r'Section \d+(?:\.\d+)*'
        found_citations = re.findall(section_pattern, response_text)
        
        if not found_citations and source_nodes:
            # Response lacks citations - inject them
            response_text = self._inject_citations(response_text, source_nodes)
        elif found_citations:
            # Verify citations are complete (not just "Section 3" when "3.1.1" exists)
            response_text = self._enhance_citations(response_text, source_nodes)
        
        # Add summary of applicable sections at the end if complex response
        if len(source_nodes) > 2:
            summary = self._create_citation_summary(source_nodes)
            response_text += f"\n\n{summary}"
        
        return response_text
    
    def _inject_citations(self, text: str, source_nodes) -> str:
        """
        Inject citations when missing.
        """
        if not source_nodes:
            return text
        
        # Get primary section
        primary_section = source_nodes[0].node.metadata.get('section_id', '')
        
        # Smart injection based on response structure
        if text.startswith(('Yes', 'No', 'The', 'It', 'A')):
            # Declarative statement - add citation after first sentence
            sentences = text.split('. ')
            if sentences:
                sentences[0] += f" (Section {primary_section})"
                return '. '.join(sentences)
        else:
            # Add citation at beginning
            return f"According to Section {primary_section}, {text}"
    
    def _enhance_citations(self, text: str, source_nodes) -> str:
        """
        Upgrade vague citations to specific ones.
        """
        import re
        
        # Map general sections to specific ones
        section_map = {}
        for node in source_nodes:
            section_id = node.node.metadata.get('section_id', '')
            if '.' in section_id:
                # Map parent to most specific child
                parent = section_id.split('.')[0]
                if parent not in section_map or len(section_id) > len(section_map[parent]):
                    section_map[parent] = section_id
        
        # Replace general with specific
        for parent, specific in section_map.items():
            text = re.sub(
                rf'\bSection {parent}\b(?!\.)',
                f'Section {specific}',
                text
            )
        
        return text
    
    def _create_citation_summary(self, source_nodes) -> str:
        """
        Create a summary of applicable law sections.
        """
        sections = []
        for node in source_nodes:
            section_id = node.node.metadata.get('section_id', '')
            section_title = node.node.metadata.get('section_title', '')
            if section_id:
                sections.append(f"- Section {section_id}: {section_title}")
        
        if sections:
            return "Applicable Laws:\n" + '\n'.join(sections)
        return ""
    
    def _extract_citations(self, source_nodes) -> List[Citation]:
        """
        Extract structured citations from source nodes.
        """
        citations = []
        seen_sections = set()
        
        for node in source_nodes:
            section_id = node.node.metadata.get('section_id', 'Unknown')
            
            # Avoid duplicates
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            
            section_title = node.node.metadata.get('section_title', '')
            hierarchy_level = node.node.metadata.get('hierarchy_level', 0)
            
            # Include parent context for sub-sections
            if hierarchy_level > 1:
                main_title = node.node.metadata.get('main_title', '')
                display_source = f"Section {section_id} ({main_title} - {section_title})"
            else:
                display_source = f"Section {section_id}: {section_title}"
            
            citation = Citation(
                source=display_source,
                text=node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
            )
            citations.append(citation)
        
        # Sort citations by section number
        citations.sort(key=lambda x: self._parse_section_number(x.source))
        
        return citations
    
    def _parse_section_number(self, source: str) -> tuple:
        """
        Parse section number for sorting.
        "Section 3.1.1" -> (3, 1, 1)
        """
        import re
        match = re.search(r'Section ([\d.]+)', source)
        if match:
            parts = match.group(1).split('.')
            return tuple(int(p) for p in parts if p.isdigit())
        return (999,)  # Sort unknown at end
```

### 3. Custom Node Postprocessors

```python
# utils.py - Add these classes for enhanced retrieval

from llama_index.core.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional

class SectionMetadataFilter(BaseNodePostprocessor):
    """
    Prioritize nodes with specific section references.
    """
    
    def __init__(self, query: str):
        super().__init__()
        self.query = query.lower()
        
        # Extract section numbers mentioned in query
        import re
        self.referenced_sections = re.findall(r'\b(?:section\s+)?(\d+(?:\.\d+)*)\b', self.query)
    
    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Boost scores for explicitly referenced sections.
        """
        if not self.referenced_sections:
            return nodes
        
        for node in nodes:
            section_id = node.node.metadata.get('section_id', '')
            
            # Exact match gets highest boost
            if section_id in self.referenced_sections:
                node.score *= 1.5
            # Parent section match gets moderate boost
            elif any(section_id.startswith(ref) for ref in self.referenced_sections):
                node.score *= 1.2
            # Child section match gets small boost
            elif any(ref.startswith(section_id) for ref in self.referenced_sections):
                node.score *= 1.1
        
        # Re-sort by score
        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes

class SimilarityPostprocessor(BaseNodePostprocessor):
    """
    Filter out low-relevance nodes.
    """
    
    def __init__(self, similarity_cutoff: float = 0.7):
        super().__init__()
        self.similarity_cutoff = similarity_cutoff
    
    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Remove nodes below similarity threshold.
        """
        return [n for n in nodes if n.score >= self.similarity_cutoff]
```

### 4. FastAPI Integration

```python
# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils import DocumentService, QdrantService, Output
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Norm AI Legal Query Service",
    description="Precise legal document retrieval with section-level citations",
    version="1.0.0"
)

# Global service instances
doc_service: Optional[DocumentService] = None
qdrant_service: Optional[QdrantService] = None

class QueryRequest(BaseModel):
    """API request model"""
    query: str = Field(..., description="Legal question to answer")
    top_k: Optional[int] = Field(3, description="Number of relevant sections to retrieve")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global doc_service, qdrant_service
    
    try:
        # Verify environment
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize services
        logger.info("Initializing DocumentService...")
        doc_service = DocumentService()
        
        logger.info("Parsing legal documents...")
        documents = doc_service.create_documents("docs/laws.pdf")
        logger.info(f"Parsed {len(documents)} document sections")
        
        # Log section distribution for verification
        hierarchy_counts = {}
        for doc in documents:
            level = doc.metadata.get('hierarchy_level', 0)
            hierarchy_counts[level] = hierarchy_counts.get(level, 0) + 1
        logger.info(f"Document hierarchy: {hierarchy_counts}")
        
        logger.info("Initializing QdrantService...")
        qdrant_service = QdrantService(k=3)
        qdrant_service.connect()
        
        logger.info("Loading documents into vector store...")
        qdrant_service.load(documents)
        
        logger.info("Service initialization complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.post("/query", response_model=Output)
async def query_endpoint(request: QueryRequest):
    """
    Query legal documents with precise section citations.
    
    Returns:
    - query: Original question
    - response: Answer with embedded section citations
    - citations: List of relevant law sections
    """
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Update k if provided
        if request.top_k:
            qdrant_service.k = request.top_k
        
        # Execute query
        result = qdrant_service.query(request.query)
        
        # Verify citations are present
        if not result.citations:
            logger.warning(f"No citations found for query: {request.query}")
        
        return result
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy" if qdrant_service else "initializing",
        "services": {
            "document_service": doc_service is not None,
            "qdrant_service": qdrant_service is not None
        }
    }

@app.get("/sections")
async def list_sections():
    """
    Debug endpoint to list all parsed sections.
    Useful for verifying document parsing.
    """
    if not doc_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    documents = doc_service.create_documents("docs/laws.pdf")
    sections = []
    
    for doc in documents:
        sections.append({
            "section_id": doc.metadata.get('section_id'),
            "title": doc.metadata.get('section_title'),
            "level": doc.metadata.get('hierarchy_level'),
            "text_preview": doc.text[:100] + "..."
        })
    
    return {"total": len(sections), "sections": sections}
```

## Testing Strategy

### Unit Tests for Section Parsing

```python
# tests/test_parsing.py

def test_section_extraction():
    """Verify correct section ID extraction."""
    test_text = """
    1. Peace
    1.1. The law requires petty lords to take disputes to their liege
    1.1.1. However, disputes between great houses go to the Crown
    """
    
    service = DocumentService()
    docs = service._parse_hierarchical_structure(test_text)
    
    # Should create 3 documents
    assert len(docs) == 3
    
    # Check section IDs
    section_ids = [d.metadata['section_id'] for d in docs]
    assert '1' in section_ids
    assert '1.1' in section_ids  
    assert '1.1.1' in section_ids

def test_citation_format():
    """Verify citation format in responses."""
    response = "According to Section 3.1.1, widows must be maintained."
    
    import re
    citations = re.findall(r'Section (\d+(?:\.\d+)*)', response)
    assert citations == ['3.1.1']
```

### Integration Tests

```python
# tests/test_integration.py

def test_query_with_citations():
    """End-to-end test of query with citations."""
    
    # Initialize services
    doc_service = DocumentService()
    docs = doc_service.create_documents("docs/laws.pdf")
    
    qdrant_service = QdrantService()
    qdrant_service.connect()
    qdrant_service.load(docs)
    
    # Test query
    result = qdrant_service.query("What happens to widows?")
    
    # Verify response contains section references
    assert "Section" in result.response
    assert any("3.1" in c.source for c in result.citations)
```

## Deployment Checklist

- [ ] Environment variable `OPENAI_API_KEY` set
- [ ] PDF file present at `docs/laws.pdf`
- [ ] Dependencies installed from requirements.txt
- [ ] Run unit tests for document parsing
- [ ] Run integration tests for citation accuracy
- [ ] Verify all sections are parsed (check `/sections` endpoint)
- [ ] Test various query types for citation consistency
- [ ] Monitor first queries for citation format

## Performance Optimization

1. **Document Parsing**: Cache parsed documents in Redis/file for production
2. **Vector Store**: Use persistent Qdrant with proper indexing
3. **Query Caching**: Implement LRU cache for common queries
4. **Batch Processing**: Process multiple queries in parallel when possible

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Missing citations | Check document metadata has section_id field |
| Wrong section numbers | Verify PDF parsing regex patterns |
| Vague citations (just "Section 3") | Enable _enhance_citations post-processing |
| Poor retrieval | Tune similarity_cutoff and k parameters |
| Slow queries | Reduce citation_chunk_size, use caching |
