import os
import pymupdf
import re
from typing import List, Dict
from llama_index.core.schema import Document


class DocumentService:
    """
    Service for parsing PDF documents into hierarchical structured documents.
    Extracts sections and subsections while maintaining precise section references.
    """
    
    def create_documents(self, file_path: str = "docs/laws.pdf") -> List[Document]:
        """
        Parse PDF file and create structured Document objects with hierarchical metadata.
        Groups related sections together for better context.
        
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
        
        # First, parse all sections
        all_sections = self._parse_all_sections(cleaned_text, raw_text)
        
        # Then group them by main section and create consolidated documents
        return self._create_consolidated_documents(all_sections)
    
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
    
    def _parse_all_sections(self, text: str, raw_text: str = "") -> Dict:
        """
        Parse all sections into a dictionary structure.
        """
        sections_data = {}
        lines = text.split('\n')
        
        current_section = None
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
                    sections_data[current_section] = {
                        'content': current_content.copy(),
                        'title': section_titles.get(current_section, "")
                    }
                
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
            sections_data[current_section] = {
                'content': current_content.copy(),
                'title': section_titles.get(current_section, "")
            }
        
        return {'sections': sections_data, 'titles': section_titles, 'raw_text': raw_text}
    
    def _create_consolidated_documents(self, all_sections: Dict) -> List[Document]:
        """
        Create consolidated documents by grouping related sections.
        """
        documents = []
        sections_data = all_sections['sections']
        section_titles = all_sections['titles']
        raw_text = all_sections['raw_text']
        
        # Group sections by main section (first number)
        main_sections = {}
        
        for section_id, section_info in sections_data.items():
            main_section_num = section_id.split('.')[0]
            
            if main_section_num not in main_sections:
                main_sections[main_section_num] = []
            
            main_sections[main_section_num].append({
                'id': section_id,
                'title': section_info['title'],
                'content': section_info['content']
            })
        
        # Create a consolidated document for each main section
        for main_section_num, sections in main_sections.items():
            # Sort sections by their ID to maintain order
            sections.sort(key=lambda x: [int(n) for n in x['id'].split('.')])
            
            # Build consolidated text that includes all subsections
            consolidated_text_parts = []
            
            # Add main section title
            main_title = section_titles.get(main_section_num, f"Section {main_section_num}")
            consolidated_text_parts.append(f"MAIN SECTION: {main_title}")
            consolidated_text_parts.append("")
            
            # Track metadata
            all_page_numbers = set()
            subsection_titles = []
            
            # Add each section with clear hierarchy
            for section in sections:
                section_id = section['id']
                hierarchy_level = len(section_id.split('.'))
                
                # Skip top-level sections that are just headers
                if hierarchy_level == 1 and len(section['content']) == 1:
                    # Check if this has subsections
                    has_subsections = any(
                        s['id'].startswith(section_id + '.') 
                        for s in sections 
                        if s['id'] != section_id
                    )
                    if has_subsections:
                        continue
                
                # Format section based on hierarchy
                indent = "  " * (hierarchy_level - 1)
                
                # Add section header with clear labeling
                consolidated_text_parts.append(f"{indent}[Section {section_id}] {section['title']}")
                
                # Add section content (skip the first item which is the title)
                content_without_title = section['content'][1:] if len(section['content']) > 1 else []
                if content_without_title:
                    content_text = ' '.join(content_without_title)
                    consolidated_text_parts.append(f"{indent}{content_text}")
                consolidated_text_parts.append("")  # Empty line for readability
                
                # Track subsection titles for metadata
                if hierarchy_level > 1:
                    subsection_titles.append(f"{section_id}: {section['title']}")
                
                # Find page number for this section
                page_num = self._find_page_number(section['content'], raw_text)
                all_page_numbers.add(page_num)
            
            # Create the consolidated text
            consolidated_text = "\n".join(consolidated_text_parts)
            
            # Build comprehensive metadata
            metadata = {
                'main_section': main_section_num,
                'main_title': main_title,
                'subsections': subsection_titles,
                'subsection_count': len(sections) - 1,  # Exclude main section if it's just a header
                'page_numbers': sorted(list(all_page_numbers)),
                'section_ids': [s['id'] for s in sections],
                # Add searchable full path for all sections
                'full_paths': [self._build_full_path(s['id'], section_titles) for s in sections]
            }
            
            # Create document
            doc = Document(
                text=consolidated_text,
                metadata=metadata,
                doc_id=f"main_section_{main_section_num}"
            )
            
            documents.append(doc)
        
        return documents
    
    def _find_page_number(self, content: List[str], raw_text: str) -> int:
        """
        Find the page number where the content appears.
        """
        page_num = 1  # Default to page 1
        
        if content and raw_text:
            # Use the first meaningful content line
            search_text = None
            for line in content:
                if len(line) > 10:  # Skip very short lines
                    search_text = line
                    break
            
            if search_text:
                text_lines = raw_text.split('\n')
                current_page = 1
                
                for line in text_lines:
                    if line.startswith('Page '):
                        try:
                            current_page = int(line.split(':')[0].replace('Page ', ''))
                        except (ValueError, IndexError):
                            pass
                    elif search_text in line:
                        page_num = current_page
                        break
        
        return page_num
    
    def _build_full_path(self, section_id: str, section_titles: dict) -> str:
        """
        Build the full hierarchical path for a section.
        """
        path_parts = []
        section_parts = section_id.split('.')
        
        for i in range(len(section_parts)):
            partial_section = '.'.join(section_parts[:i+1])
            title = section_titles.get(partial_section, f"Section {partial_section}")
            path_parts.append(f"{partial_section}. {title}")
        
        return " > ".join(path_parts)
