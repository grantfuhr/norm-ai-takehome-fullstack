#!/usr/bin/env python3
"""
Script to extract raw text from PDF using PyMuPDF.
This preserves the exact text as extracted without any processing,
useful as input for testing document parsing logic.
"""

import pymupdf

def extract_pdf_text(pdf_path: str, output_path: str) -> None:
    """
    Extract raw text from PDF file and save to text file.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Path where to save the extracted text
    """
    raw_text = ''

    # Read the PDF and extract raw text
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        raw_text += f'Page {page_num + 1}:\n{page_text}\n\n'
    
    doc.close()

    # Save the raw extracted text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(raw_text)

    print(f'Raw PDF text extracted and saved to {output_path}')
    print(f'Total characters: {len(raw_text)}')
    print(f'Total pages processed: {total_pages}')

if __name__ == "__main__":
    # Extract text from the laws PDF
    extract_pdf_text('docs/laws.pdf', 'docs/raw_pdf_text.txt')