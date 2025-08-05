
# HackRx 6.0 - Intelligent Query-Retrieval System for Insurance Policies

## Overview

This project implements a RESTful API service to answer complex natural language queries about insurance policies by analyzing large policy documents (PDFs). It utilizes a Retrieval-Augmented Generation (RAG) approach combining:

- Document ingestion and processing (PDF parsing and text chunking)
- Semantic search with embeddings and FAISS vector index
- Large Language Model API calls (Google Gemini or GPT-4) for reasoning and explanation
- Secure API with Bearer token authentication

## Features

- Accepts policy documents as PDF uploads or URLs
- Extracts structured and unstructured text for semantic indexing
- Answers multiple user questions in one request
- Provides references to exact policy clauses used in answers
- Returns structured JSON responses
- Enforces secure authentication via token
- Designed to meet HackRx 6.0 submission requirements

## API Endpoint

- Path: `/hackrx/run`
- Method: `POST`
- Authentication: Bearer token (set in header `Authorization: Bearer `)
- Content-Type: `multipart/form-data`
- Request parameters:

  | Parameter       | Type           | Description                          |
  | --------------- | -------------- | ---------------------------------- |
  | `pdf_file`      | file (PDF)     | The insurance policy PDF document  |
  | `questions_json`| string (JSON)  | JSON string with `{"questions": ["Q1", "Q2", ...]}` |

- Response format (JSON):
  
  ```
  {
    "answers": [
      "Answer to question 1",
      "Answer to question 2",
      ...
    ]
  }
  ```

## Getting Started

### Prerequisites

- Python 3.9+
- API key for LLM service (Google Gemini API or OpenAI GPT)
- Install dependencies (see `requirements.txt`)

### Installation

```
git clone 
cd 
pip install -r requirements.txt
```

### Running Locally

Set environment variables for API keys and token:

```
export GEMINI_API_KEY=
export API_TOKEN=
```

Start the FastAPI server:

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing with Client

Use the provided `client.py` to interact with your API from the command line:

```
python client.py
```

It will prompt for a PDF file path and allow you to enter queries interactively.

## System Architecture

1. **PDF Parsing:** Reads and extracts text from uploaded PDF.
2. **Text Chunking:** Divides document into manageable chunks with overlaps.
3. **Embedding & Indexing:** Uses SentenceTransformers to embed chunks; builds FAISS index.
4. **Semantic Retrieval:** Embeds the query, retrieves most relevant chunks.
5. **LLM Reasoning:** Sends retrieved chunks + query to LLM API, generates precise answers.
6. **Response Formation:** Compiles structured answers with references.

## Security

- All requests require Bearer token authentication.
- API token checked before processing.

## Performance Considerations

- FAISS vector search ensures fast similarity lookup.
- Caching or persistent storage can be added to reuse embeddings (not implemented here).
- Respect API rate limits when interacting with external LLM services.

## License

This project is submitted for HackRx 6.0 as per contest rules.

## Acknowledgements

- Google Gemini API for LLM capabilities.
- SentenceTransformers and FAISS for semantic search.
- FastAPI for API framework.

## Contact
    dharsansp254@gmail.com
    