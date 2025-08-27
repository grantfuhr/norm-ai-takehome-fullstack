import os
import pymupdf
import re
from typing import List
from llama_index.core.schema import Document


class DocumentService:
    """
    Service for parsing PDF documents into hierarchical structured documents.
    Extracts sections and subsections while maintaining precise section references.
    """
    
    def create_documents(self, file_path: str = "docs/laws.pdf") -> List[Document]:
        """
        Parse PDF file and create structured Document objects with hierarchical metadata.
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            List of Document objects with section metadata
        """
        # Extract raw text from PDF
        raw_text = ''

        # Read the PDF and extract raw text
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()
            raw_text += f'Page {page_num + 1}:\n{page_text}\n\n'
        
        doc.close()
        
        # Clean and parse the text
        cleaned_text = self._clean_pdf_text(raw_text)
        return self._parse_hierarchical_structure(cleaned_text, raw_text)
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF text by rejoining words that were split during extraction.
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and page markers
            if not line or line.startswith('Page '):
                i += 1
                continue
                
            # Check if this looks like a section number pattern
            if re.match(r'^\d+(\.\d+)*\.$', line):
                # This is a section number, keep it separate
                cleaned_lines.append(line)
                i += 1
                continue
            
            # For other lines, try to reconstruct sentences by joining words
            current_sentence = []
            
            # Collect words until we hit a section number or end
            while i < len(lines) and lines[i].strip():
                word = lines[i].strip()
                
                # If we hit a section number pattern, stop collecting
                if re.match(r'^\d+(\.\d+)*\.$', word):
                    break
                    
                current_sentence.append(word)
                i += 1
            
            # Join the collected words into a sentence
            if current_sentence:
                cleaned_lines.append(' '.join(current_sentence))
        
        return '\n'.join(cleaned_lines)
    
    def _parse_hierarchical_structure(self, text: str, raw_text: str = "") -> List[Document]:
        """
        Parse the cleaned text into hierarchical document structure.
        """
        documents = []
        lines = text.split('\n')
        
        current_section = None
        current_subsection = None
        current_content = []
        section_titles = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check for section patterns
            section_match = re.match(r'^(\d+(\.\d+)*)\.$', line)
            
            if section_match:
                # Save previous section if exists
                if current_section and current_content:
                    self._create_section_document(
                        documents, current_section, current_content, 
                        section_titles, lines, raw_text
                    )
                
                # Start new section
                section_num = section_match.group(1)
                current_section = section_num
                current_content = []
                
                # Get the title (next non-empty line)
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                
                if i < len(lines):
                    title = lines[i].strip()
                    section_titles[section_num] = title
                    current_content.append(title)
                
            else:
                # This is content, add to current section
                if current_section:
                    current_content.append(line)
            
            i += 1
        
        # Handle the last section
        if current_section and current_content:
            self._create_section_document(
                documents, current_section, current_content, 
                section_titles, lines, raw_text
            )
        
        return documents
    
    def _create_section_document(self, documents: List[Document], section_id: str, 
                                content: List[str], section_titles: dict, 
                                 all_lines: List[str], raw_text: str = ""):
        """
        Create a Document object for a section with proper metadata.
        """
        # Join content and clean up
        full_text = ' '.join(content)
        
        # Determine hierarchy level
        hierarchy_level = len(section_id.split('.'))
        
        # Skip creating documents for top-level sections that only contain a title
        # and have subsections (these are just headers)
        if hierarchy_level == 1 and len(content) == 1:
            # Check if this section has any subsections by looking through all lines
            has_subsections = False
            for line in all_lines:
                line = line.strip()
                # Look for section patterns that are subsections of current section
                section_match = re.match(r'^(\d+(\.\d+)*)\.', line)
                if section_match:
                    found_section = section_match.group(1)
                    if found_section.startswith(section_id + '.'):
                        has_subsections = True
                        break
            
            # Only skip if it has subsections (meaning it's just a header)
            if has_subsections:
                return
        
        # Get section title
        section_title = section_titles.get(section_id, content[0] if content else "")
        
        # Determine page number based on page markers in the text
        page_num = 1  # Default to page 1
        if content:
            # Look for the section content in the original raw text to find its page
            first_content = content[0]
            text_lines = raw_text.split('\n') if raw_text else []
            
            current_page = 1
            for line in text_lines:
                if line.startswith('Page '):
                    try:
                        current_page = int(line.split(':')[0].replace('Page ', ''))
                    except (ValueError, IndexError):
                        pass
                elif first_content in line:
                    page_num = current_page
                    break
        
        # Create parent section reference
        parent_section = None
        if hierarchy_level > 1:
            parts = section_id.split('.')
            parent_section = '.'.join(parts[:-1])
        
        # Get main section for context
        main_section = section_id.split('.')[0]
        main_title = section_titles.get(main_section, "")
        
        # Create full path for context
        path_parts = []
        section_parts = section_id.split('.')
        for i in range(len(section_parts)):
            partial_section = '.'.join(section_parts[:i+1])
            title = section_titles.get(partial_section, f"Section {partial_section}")
            path_parts.append(f"{partial_section}. {title}")
        
        full_path = " > ".join(path_parts)
        
        # Create document
        document = Document(
            text=full_text,
            metadata={
                'section_id': section_id,
                'section_title': section_title,
                'parent_section': parent_section,
                'main_section': main_section,
                'main_title': main_title,
                'hierarchy_level': hierarchy_level,
                'full_path': full_path,
                'page_number': page_num
            }
        )
        
        documents.append(document)

