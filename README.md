# Future improvements/goals
1. Scale number of texts to 100, 500, 1000 without losing accuracy or efficiency
2. Persist Chroma vector database to prevent costly recomputaitons
3. Build working frontend
4. Cross-paper examination
5. Research gap identification: findings, strengths, limitations, stated future work

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

```mermaid
flowchart TD

    subgraph INDEXING["Document Indexing"]
        A["Preprocessed arXiv Text"]
        B["Split into Chunks<br/>500 tokens, 50-token overlap"]
        C["Embedding Model"]
        D[("Chroma Vector Store")]

        A --> B
        B --> C
        C --> D
    end

    subgraph QUERY["Query & Retrieval"]
        E["User Query"]
        F["Convert Query to Embedding"]
        G["Retriever"]
        H["Relevant Chunks + Metadata"]

        E --> F
        F --> G
        D --> G
        G --> H
    end

    subgraph GENERATION["Answer Generation"]
        I["QA Chain"]
        J["GPT-4<br/>Query + Retrieved Context"]
        K["Grounded Answer<br/>+ Sources"]

        H --> I
        E --> I
        I --> J
        J --> K
    end


Model Reliability Note

Smaller or weaker language models may misinterpret retrieved chunks, give too much weight to less relevant evidence, or make claims that are not fully supported by the provided text. RAG reduces hallucination by grounding the model in retrieved sources, but it does not eliminate hallucinations.

Testing has confirmed that models like gpt-nano and deepseek-v4-flash extract data from the wrong chunks. Currently the model is set to use gpt-4.