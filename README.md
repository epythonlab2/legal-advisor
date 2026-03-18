# Ethio Legal Assistant

AI-powered legal question assistant for Ethiopian laws using Large
Language Models (LLMs) and Retrieval Augmented Generation (RAG).

## Project Vision

Legal information in Ethiopia is often difficult to access, scattered
across PDFs, and hard to search.\
This project builds a local AI assistant that allows users to ask legal
questions in natural language and receive answers grounded in Ethiopian
legal documents.

Example questions:

-   What are employee rights under Ethiopian labor law?
-   What is the legal process for land disputes?
-   What penalties exist for contract violations?

------------------------------------------------------------------------

## Features

-   Natural language legal Q&A
-   Retrieval from Ethiopian legal documents
-   Local LLM inference (CPU friendly)
-   Source citations from legal texts
-   Modular architecture

Planned:

-   Amharic language support
-   Voice interface
-   Legal document summarization
-   Case law search

------------------------------------------------------------------------

## Architecture

User Question\
↓\
Embedding Model\
↓\
Vector Database Search\
↓\
Relevant Legal Documents\
↓\
LLM Generates Answer\
↓\
Response with Sources

Main components:

1.  Document processing pipeline
2.  Vector database
3.  Embedding model
4.  Local LLM
5.  API / Interface

------------------------------------------------------------------------

## Technology Stack

-   Python
-   LangChain
-   FAISS Vector Database
-   SentenceTransformers
-   Local LLM (Mistral / Llama / Phi)
-   FastAPI
-   Streamlit

Optional tools:

-   Ollama
-   HuggingFace Transformers
-   ChromaDB

------------------------------------------------------------------------

## Project Structure

    ethio-legal-assistant
    │
    ├── data
    │   ├── raw_legal_docs
    │   └── processed_docs
    │
    ├── embeddings
    │   └── build_vector_db.py
    │
    ├── rag
    │   ├── retriever.py
    │   ├── prompt_template.py
    │   └── pipeline.py
    │
    ├── models
    │   └── load_llm.py
    │
    ├── api
    │   └── main.py
    │
    ├── app
    │   └── streamlit_app.py
    │
    ├── notebooks
    │   └── experiments.ipynb
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Step 1 --- Collect Ethiopian Legal Documents

Gather public legal documents such as:

-   Ethiopian Constitution
-   Labor Proclamation
-   Civil Code
-   Criminal Code
-   Land laws

Save them in:

    data/raw_legal_docs/

------------------------------------------------------------------------

## Step 2 --- Install Environment

Create environment:

    python -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Example dependencies:

-   langchain
-   sentence-transformers
-   faiss-cpu
-   transformers
-   accelerate
-   pypdf
-   fastapi
-   uvicorn
-   streamlit

------------------------------------------------------------------------

## Step 3 --- Process Legal Documents

Extract text and split into chunks.

    python data/process_documents.py

This step performs:

-   PDF extraction
-   text cleaning
-   chunking legal documents

------------------------------------------------------------------------

## Step 4 --- Create Embeddings

Generate vector embeddings.

    python embeddings/build_vector_db.py

Result:

    vector_store/

------------------------------------------------------------------------

## Step 5 --- Load Local LLM

Recommended models:

-   Mistral 7B
-   Llama 3
-   Phi-2

Example loader:

    models/load_llm.py

------------------------------------------------------------------------

## Step 6 --- Build RAG Pipeline

Pipeline steps:

1.  Question embedding
2.  Vector similarity search
3.  Retrieve relevant documents
4.  Generate answer using LLM

Main file:

    rag/pipeline.py

------------------------------------------------------------------------

## Step 7 --- Run API

    uvicorn api.main:app --reload

Example endpoint:

    POST /ask

Request example:

    {
     "question": "What rights do workers have under Ethiopian labor law?"
    }

------------------------------------------------------------------------

## Step 8 --- Launch Interface

    streamlit run app/streamlit_app.py

Workflow:

User asks question\
System retrieves legal documents\
LLM generates response\
Sources are shown

------------------------------------------------------------------------

## Example Interaction

Question:

    What is the probation period allowed under Ethiopian labor law?

Response:

Explanation of law\
Relevant legal clause\
Source citation

------------------------------------------------------------------------

## Evaluation Strategy

-   Test with curated legal questions
-   Measure hallucination rate
-   Improve prompts and retrieval
-   Verify answers with citations

------------------------------------------------------------------------

## Future Improvements

-   Amharic LLM support
-   Telegram / WhatsApp chatbot
-   Ethiopian legal dataset expansion
-   Court case retrieval

------------------------------------------------------------------------

## Ethical Notice

This project provides informational assistance only and does not replace
professional legal advice.

Users should consult licensed legal professionals for official legal
decisions.

------------------------------------------------------------------------

## Author

**Asibeh Tenager**\
AI Educator & Developer

Focus:

-   Machine Learning
-   LLM Systems
-   AI for African solutions