# AI Research Assistant - RAG System

An AI-powered research assistant that allows users to ask questions about scientific papers using Retrieval-Augmented Generation (RAG).

The system processes PDF research papers, retrieves relevant information using vector search, and generates answers using a local Large Language Model.

The goal of this project is not only to build a chatbot, but to create an explainable and evaluatable RAG pipeline suitable for research and portfolio demonstration.

---

# Project Overview

This project implements a complete Retrieval-Augmented Generation pipeline for scientific document question answering.

The system allows users to upload research papers and ask questions about their content. Relevant information is retrieved from the document and provided to a local LLM to generate grounded answers.

---

# Architecture

```
PDF Paper
    |
    v
PDF Loading
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
FAISS Vector Database
    |
    v
Retriever
    |
    v
Relevant Context
    |
    v
Local LLM (Gemma2)
    |
    v
Generated Answer
    |
    v
Evaluation
```

---

# Technologies

- Python
- LangChain
- FAISS
- Sentence Transformers
- HuggingFace Embeddings
- Ollama
- Gemma2:2b
- NLTK

---

# Current Dataset

The current benchmark uses:

**Attention Is All You Need**

The system evaluates questions related to:

- Transformer architecture
- Multi-head attention
- Machine translation tasks
- Main contributions of the paper

---

# RAG Pipeline

The current pipeline includes:

## 1. Document Processing

- Loading PDF research papers
- Extracting text content
- Splitting documents into smaller chunks

## 2. Embedding Generation

Each chunk is converted into a vector representation using:

```
sentence-transformers/all-MiniLM-L6-v2
```

## 3. Vector Database

FAISS is used to store and retrieve document embeddings efficiently.

## 4. Retrieval

The system uses:

```
Max Marginal Relevance (MMR)
```

to retrieve relevant document chunks while maintaining diversity in retrieved context.

## 5. Answer Generation

A local LLM is used through Ollama:

```
Gemma2:2b
```

The model generates answers based only on retrieved document context to reduce hallucination.

---

# Evaluation Pipeline

A dedicated evaluation pipeline is implemented to measure RAG performance.

The evaluation system currently includes two metrics.

---

## 1. Keyword-Based Evaluation

The generated answer is compared against expected keywords.

Example:

Expected keywords:

```
Transformer
attention
convolution
```

The evaluator checks whether these concepts appear in the generated response.

To improve robustness, stemming-based normalization is applied.

Example:

```
recurrent
recurrence
```

are converted into similar root representations.

---

## 2. Semantic Similarity Evaluation

The generated answer and expected answer are converted into embeddings.

Cosine similarity is used to measure semantic similarity between the two answers.

This helps evaluate cases where the wording is different but the meaning is similar.

Current evaluation metrics:

- Keyword Accuracy
- Average Semantic Similarity

---

# Project Structure

```
AI-Research-Assistant/

├── data/
│   └── papers/
│       └── attention.pdf
│
├── evaluation/
│   ├── evaluate_rag.py
│   └── questions.json
│
├── models/
│   └── faiss_index/
│
├── notebooks/
│   ├── 01_rag_concept.ipynb
│   └── 02_similarity_search.ipynb
│
├── src/
│   ├── build_vector_db.py
│   ├── config.py
│   ├── debug_chunks.py
│   ├── faiss_store.py
│   ├── pdf_loader.py
│   ├── query_faiss.py
│   ├── rag_pipeline.py
│   └── text_splitter.py
│
├── tests/
│
├── README.md
└── requirements.txt
```

---

# Current Results

The current internal benchmark evaluates the RAG system using:

- Keyword Accuracy
- Semantic Similarity

Current evaluation setup:

- Dataset: Attention Is All You Need paper
- Number of evaluation questions: 3

The benchmark is designed for development evaluation and continuous improvement.

---

# Development Process

The project follows an experiment-driven development workflow.

Each improvement is evaluated by:

1. Identifying a limitation
2. Changing one component
3. Running evaluation
4. Comparing results
5. Recording the change through Git commits

Examples of implemented improvements:

- Centralized embedding configuration
- Metadata-aware document processing
- Improved keyword normalization using stemming
- Added semantic answer evaluation

---

# Future Improvements

Planned improvements:

- Retrieval evaluation metrics:
  - Recall@K
  - Precision@K

- Larger evaluation datasets

- Improved document filtering

- Advanced RAG evaluation frameworks

- Comparison between different retrieval strategies

- LLM comparison experiments

---

# Notes

This project is continuously developed as a research-oriented RAG system.

The main focus is improving:

- Retrieval quality
- Answer reliability
- Evaluation methodology
- Explainability of generated responses