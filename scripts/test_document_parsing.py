#!/usr/bin/env python3
"""
Test script to examine the output of DocumentService.create_documents()
and save a representation of the parsed documents.
"""

import json
from app.document import DocumentService

def test_document_parsing():
    """Test document parsing and save representation."""
    service = DocumentService()
    documents = service.create_documents("docs/laws.pdf")
    
    print(f"Total documents created: {len(documents)}")
    
    # Create a representation of the documents
    doc_representation = []
    
    for i, doc in enumerate(documents):
        doc_dict = {
            "index": i,
            "text": doc.text,
            "metadata": doc.metadata,
            "text_length": len(doc.text)
        }
        doc_representation.append(doc_dict)
        
        # Print summary
        section_id = doc.metadata.get('section_id', 'Unknown')
        section_title = doc.metadata.get('section_title', 'No Title')
        hierarchy_level = doc.metadata.get('hierarchy_level', 0)
        print(f"Doc {i}: Section {section_id} - {section_title} (Level {hierarchy_level})")
        print(f"  Text length: {len(doc.text)} chars")
        print(f"  Full path: {doc.metadata.get('full_path', 'N/A')}")
        print()
    
    # Save to JSON file
    with open('parsed_documents_output.json', 'w', encoding='utf-8') as f:
        json.dump(doc_representation, f, indent=2, ensure_ascii=False)
    
    print(f"Document representation saved to 'parsed_documents_output.json'")

if __name__ == "__main__":
    test_document_parsing()
