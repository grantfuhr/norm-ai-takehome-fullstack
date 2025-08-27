"""
Unit tests for DocumentService PDF parsing functionality.
Tests the actual PDF parsing behavior using the real laws.pdf file.
Uses a standalone implementation to avoid import issues.
"""

import pytest
import os
import sys
from pathlib import Path
import pymupdf
import re
from typing import List

from llama_index.core.schema import Document
from app.document import DocumentService

class TestDocumentParsing:
    """Test suite for DocumentService PDF parsing functionality."""
    
    @pytest.fixture
    def document_service(self):
        """Create a DocumentService instance for testing."""
        return DocumentService()
    
    @pytest.fixture
    def pdf_path(self):
        """Path to the test PDF file."""
        return "docs/laws.pdf"
    
    @pytest.fixture
    def documents(self, document_service, pdf_path):
        """Parse documents from the PDF for use in tests."""
        return document_service.create_documents(pdf_path)
    
    def test_pdf_file_exists(self, pdf_path):
        """Test that the PDF file exists and is accessible."""
        assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
        assert os.path.getsize(pdf_path) > 0, "PDF file is empty"
    
    def test_documents_creation(self, documents):
        """Test that documents are successfully created from PDF."""
        assert isinstance(documents, list), "create_documents should return a list"
        assert len(documents) > 0, "Should create at least one document"
        assert all(isinstance(doc, Document) for doc in documents), "All items should be Document objects"
    
    def test_expected_document_count(self, documents):
        """Test that the expected number of documents are created."""
        expected_count = 44
        assert len(documents) == expected_count, f"Expected {expected_count} documents, got {len(documents)}"
    
    def test_hierarchy_levels_distribution(self, documents):
        """Test that documents are distributed across expected hierarchy levels."""
        level_counts = {}
        for doc in documents:
            level = doc.metadata.get('hierarchy_level', 0)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        expected_distribution = {
            1: 11,  # Main sections
            2: 15,  # Subsections  
            3: 13,  # Sub-subsections
            4: 5    # Deep nested
        }
        
        for level, expected_count in expected_distribution.items():
            actual_count = level_counts.get(level, 0)
            assert actual_count == expected_count, \
                f"Level {level}: expected {expected_count} documents, got {actual_count}"
    
    def test_main_sections_exist(self, documents):
        """Test that all expected main sections are parsed."""
        main_sections = [doc for doc in documents if doc.metadata.get('hierarchy_level') == 1]
        section_ids = [doc.metadata.get('section_id') for doc in main_sections]
        
        expected_main_sections = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        
        for section_id in expected_main_sections:
            assert section_id in section_ids, f"Main section {section_id} not found"
    
    def test_main_section_titles(self, documents):
        """Test that main sections have correct titles."""
        main_sections = {doc.metadata.get('section_id'): doc.metadata.get('section_title') 
                        for doc in documents if doc.metadata.get('hierarchy_level') == 1}
        
        expected_titles = {
            '1': 'Peace',
            '2': 'Religion', 
            '3': 'Widows',
            '4': 'Trials',
            '5': 'Taxes',
            '6': 'Thievery',
            '7': 'Poaching',
            '8': 'Outlawry',
            '9': 'Slavery',
            '10': 'Watch',
            '11': 'Baking'
        }
        
        for section_id, expected_title in expected_titles.items():
            assert section_id in main_sections, f"Section {section_id} not found"
            actual_title = main_sections[section_id]
            assert actual_title == expected_title, \
                f"Section {section_id}: expected '{expected_title}', got '{actual_title}'"
    
    def test_subsection_parent_relationships(self, documents):
        """Test that subsections correctly reference their parent sections."""
        subsections = [doc for doc in documents if doc.metadata.get('hierarchy_level') == 2]
        
        for doc in subsections:
            section_id = doc.metadata.get('section_id')
            parent_section = doc.metadata.get('parent_section')
            
            expected_parent = section_id.split('.')[0]
            assert parent_section == expected_parent, \
                f"Section {section_id}: expected parent '{expected_parent}', got '{parent_section}'"
    
    def test_deep_nested_sections(self, documents):
        """Test that deeply nested sections are parsed correctly."""
        deep_sections = [doc for doc in documents if doc.metadata.get('hierarchy_level') == 4]
        section_ids = [doc.metadata.get('section_id') for doc in deep_sections]
        
        expected_deep_sections = ['5.1.2.1', '10.1.1.1', '10.1.1.2', '10.1.1.3', '10.1.1.4']
        
        for section_id in expected_deep_sections:
            assert section_id in section_ids, f"Deep nested section {section_id} not found"
    
    def test_section_text_content(self, documents):
        """Test that sections contain expected text content."""
        section_3_1_1 = next((doc for doc in documents 
                             if doc.metadata.get('section_id') == '3.1.1'), None)
        assert section_3_1_1 is not None, "Section 3.1.1 not found"
        
        text = section_3_1_1.text.lower()
        assert 'widow' in text, "Section 3.1.1 should contain 'widow'"
        assert 'maintain' in text, "Section 3.1.1 should contain 'maintain'"
        assert 'heir' in text, "Section 3.1.1 should contain 'heir'"
    
    def test_metadata_completeness(self, documents):
        """Test that all documents have complete metadata."""
        required_metadata_fields = [
            'section_id', 'section_title', 'hierarchy_level', 
            'main_section', 'full_path', 'page_number'
        ]
        
        for i, doc in enumerate(documents):
            for field in required_metadata_fields:
                assert field in doc.metadata, \
                    f"Document {i} missing required metadata field: {field}"
                assert doc.metadata[field] is not None, \
                    f"Document {i} has None value for metadata field: {field}"
    
    def test_section_id_format(self, documents):
        """Test that section IDs follow expected format."""
        section_id_pattern = re.compile(r'^\d+(\.\d+)*$')
        
        for doc in documents:
            section_id = doc.metadata.get('section_id')
            assert section_id_pattern.match(section_id), \
                f"Section ID '{section_id}' doesn't match expected format"
    
    def test_specific_law_content(self, documents):
        """Test parsing of specific law content."""
        # Test theft punishment section
        section_6_1 = next((doc for doc in documents 
                           if doc.metadata.get('section_id') == '6.1'), None)
        assert section_6_1 is not None, "Section 6.1 not found"
        
        text = section_6_1.text.lower()
        assert 'thief' in text, "Section 6.1 should mention thief"
        assert 'finger' in text or 'hand' in text, "Section 6.1 should mention punishment"
        
        # Test trial by combat section  
        section_4_2_1 = next((doc for doc in documents 
                             if doc.metadata.get('section_id') == '4.2.1'), None)
        assert section_4_2_1 is not None, "Section 4.2.1 not found"
        
        text = section_4_2_1.text.lower()
        assert 'knight' in text, "Section 4.2.1 should mention knight"
        assert 'combat' in text, "Section 4.2.1 should mention combat"
    
    def test_no_duplicate_sections(self, documents):
        """Test that there are no duplicate section IDs."""
        section_ids = [doc.metadata.get('section_id') for doc in documents]
        unique_section_ids = set(section_ids)
        
        assert len(section_ids) == len(unique_section_ids), \
            f"Found duplicate section IDs: {len(section_ids)} total, {len(unique_section_ids)} unique"
    
    def test_nonexistent_file(self):
        """Test behavior when PDF file doesn't exist."""
        doc_service = DocumentService()
        
        with pytest.raises(pymupdf.FileNotFoundError):
            doc_service.create_documents("nonexistent.pdf")
    
    def test_text_cleaning_edge_cases(self):
        """Test text cleaning with edge case inputs."""
        doc_service = DocumentService()
        
        # Test empty string
        result = doc_service._clean_pdf_text("")
        assert result == "", "Empty string should return empty string"
        
        # Test string with only whitespace
        result = doc_service._clean_pdf_text("   \n\n   \n")
        assert result.strip() == "", "Whitespace-only string should return empty when stripped"
