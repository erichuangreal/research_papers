# Goal
The purpose of the data scraper is to preprocess and clean data from research papers to prepare for model training.

# Data collection and preprocessing
1. **Paper Collection**\
Research papers are collected from arXiv.org based on a selected topic or category. Paper metadata, such as the title, authors, abstract, publication date, and arXiv ID, is stored alongside each document.
2. **Text Extraction**\
PyMuPDF is used as the primary extraction method because most arXiv PDFs contain embedded text. If the extracted text is missing, corrupted, or of poor quality, Surya OCR is used as a fallback.
3. **Cleaning the data**\
The extracted data is processed to remove:
- HTML artifacts
- dupes using minhash
- PIIs
- repetitive N-grams
4. **Data output**\
Output will be ready for model training in structure JSON format

# RAG component
Performed with preprocessed texts and metadata.

Preprocessed text chunks
(500 tokens, 50-token overlap)
        ↓
Embedding model
        ↓
Vector store (Chroma)
        ↓
User query
        ↓
Query converted into embedding
        ↓
Retriever searches Chroma
        ↓
Most relevant text chunks + metadata
        ↓
QA Chain
        ↓
LLM (gpt-4) receives query + retrieved chunks
        ↓
Answer

