# DocumentService Unit Tests

This directory contains comprehensive unit tests for the DocumentService PDF parsing functionality.

## Test Files

### `test_document_parsing.py`
Main test suite that validates the PDF parsing functionality using the actual `docs/laws.pdf` file.

## Test Coverage

The tests verify the following aspects of DocumentService:

### 📄 **PDF Processing**
- ✅ PDF file accessibility and non-empty content
- ✅ Successful document creation from PDF
- ✅ Correct document count (44 expected documents)

### 🏗️ **Hierarchical Structure**
- ✅ Proper distribution across hierarchy levels:
  - Level 1: 11 main sections (1. Peace, 2. Religion, etc.)
  - Level 2: 15 subsections (1.1, 2.1, etc.)  
  - Level 3: 13 sub-subsections (3.1.1, 4.1.1, etc.)
  - Level 4: 5 deeply nested (5.1.2.1, 10.1.1.1, etc.)
- ✅ All 11 main sections exist (1-11)
- ✅ Correct main section titles (Peace, Religion, Widows, etc.)
- ✅ Parent-child relationships are accurate
- ✅ Deep nested sections parsed correctly

### 🏷️ **Metadata Validation**
- ✅ Complete metadata for all documents:
  - `section_id`: Precise section number (e.g., "3.1.1")
  - `section_title`: Descriptive title
  - `hierarchy_level`: Depth in structure  
  - `parent_section`: Parent section reference
  - `main_section`: Top-level context
  - `full_path`: Complete hierarchical path
  - `page_number`: Page location
- ✅ Section ID format validation (regex: `^\d+(\.\d+)*$`)
- ✅ No duplicate section IDs

### 📝 **Content Accuracy**
- ✅ Specific law content verification:
  - Section 3.1.1 contains widow/maintain/heir keywords
  - Section 6.1 contains thief/punishment keywords  
  - Section 4.2.1 contains knight/combat keywords
- ✅ Text cleaning removes excessive whitespace
- ✅ No empty document content

### 🛡️ **Error Handling**
- ✅ Proper FileNotFoundError for missing files
- ✅ Edge case handling for empty/whitespace input

## Running Tests

```bash
# Run all DocumentService tests
uv run pytest tests/test_document_parsing.py -v

# Run specific test
uv run pytest tests/test_document_parsing.py::TestDocumentParsing::test_main_section_titles -v

# Run with coverage
uv run pytest tests/test_document_parsing.py --cov=app.utils
```

## Test Data

Tests use the actual PDF file at `docs/laws.pdf` containing Game of Thrones laws with the following structure:

```
1. Peace
  1.1. [Petty lords disputes]
2. Religion  
  2.1. [Holy men arms prohibition]
3. Widows
  3.1. [Widow's Law]
    3.1.1. [Heir maintenance requirements]
    3.1.2. [Castle protection]
    3.1.3. [Disinheritance prevention]
4. Trials
  4.1. [Crown trials]
    4.1.1. [Trial procedures]
  4.2. [Combat trials]
    4.2.1. [Knight combat rights]
    4.2.2. [Champion rules]
    4.2.3. [Trial of seven]
    4.2.4. [Royalty rules]
5. Taxes
  5.1. [Local collection]
    5.1.1. [Lords to crown]
    5.1.2. [Great Houses]
      5.1.2.1. [Crown payment exceptions]
  5.2. [Treasurers]
6. Thievery
  6.1. [Finger/hand punishment]
  6.2. [Pickpockets]
  6.3. [Sept theft]
7. Poaching
  7.1. [Punishment options]
8. Outlawry
  8.1. [Death by hanging]
9. Slavery
  9.1. [Illegal status]
    9.1.1. [Religious prohibition]
    9.1.2. [Selling punishment]
10. Watch
  10.1. [Alternative punishment]
    10.1.1. [Who can join]
      10.1.1.1. [Vow requirements]
      10.1.1.2. [Oath breaking]
      10.1.1.3. [Refusing orders]
      10.1.1.4. [Women exclusion]
11. Baking
  11.1. [Sawdust flour punishment]
```

## Test Philosophy

These tests follow the requirement to use the **actual PDF** rather than mocks, ensuring that:

1. **Real-world accuracy**: Tests validate against actual PDF extraction behavior
2. **Regression detection**: Changes to PDF parsing logic are immediately caught
3. **Structure verification**: Hierarchical relationships are validated with real data
4. **Content validation**: Specific legal content is verified for accuracy

The raw PDF text is preserved in `docs/raw_pdf_text.txt` for reference and additional testing scenarios.
