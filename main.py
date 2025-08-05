# main.py
from fastapi import Depends, FastAPI, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os # For environment variables
import json # Needed for JSON parsing of Gemini API response
import asyncio # Needed for await in call_gemini_api
from io import BytesIO # Needed for pdfplumber to read bytes

# --- FastAPI Initialization ---
app = FastAPI(
    title="HackRx RAG Policy Assistant",
    description="An API to answer questions based on a provided PDF policy document using RAG.",
    version="1.0.0"
)

# --- Security Configuration ---
# Load API_TOKEN from environment variable. Provide a default for local testing IF NECESSARY,
# but for production, ensure it's always set via environment.
API_TOKEN = os.getenv("API_TOKEN") 
if not API_TOKEN:
    # Fallback for local development if not set, but warn
    print("WARNING: API_TOKEN environment variable not set. Using a default for development.")
    API_TOKEN = "your-super-secret-development-token" # CHANGE THIS FOR PRODUCTION!

bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Dependency to verify the Bearer token in the Authorization header.
    """
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- PDF Parsing Helper ---
def parse_pdf_from_bytes(pdf_bytes: bytes) -> str | None:
    """
    Parses PDF content from bytes and extracts all text.
    Returns the extracted text as a single string, or None if an error occurs.
    """
    try:
        print("Attempting to parse PDF from bytes.")
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print("PDF text extracted from bytes.")
        return text
    except pdfplumber.PDFSyntaxError as e:
        print(f"Error parsing PDF syntax from bytes: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during PDF parsing from bytes: {e}")
        return None

# --- Text Chunking Helper ---
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits a long string into smaller, overlapping chunks.
    Ensures chunks are not empty.
    """
    chunks = []
    if not text:
        return chunks
        
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if chunk_size <= overlap and start < len(text) and not chunk:
            break
    print(f"Text chunked into {len(chunks)} pieces.")
    return chunks

# --- RAG System Class (Embeddings & Semantic Search) ---
class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []
        print(f"SentenceTransformer model '{model_name}' loaded.")

    def create_index(self, text_chunks: List[str]):
        """
        Encodes text chunks and builds a FAISS index.
        """
        if not text_chunks:
            self.index = None
            self.text_chunks = []
            print("No text chunks provided, FAISS index not created.")
            return

        self.text_chunks = text_chunks
        embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        print(f"FAISS index created with {self.index.ntotal} vectors.")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Finds the top-k most relevant chunks for a given query using the FAISS index.
        Returns a list of relevant text chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            print("FAISS index is not initialized or empty. Cannot perform search.")
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, k)
        
        relevant_chunks = []
        for i in I[0]:
            if 0 <= i < len(self.text_chunks):
                relevant_chunks.append(self.text_chunks[i])
        
        print(f"Found {len(relevant_chunks)} relevant chunks for the query.")
        return relevant_chunks

# --- Global RAG System Instance ---
rag_system = RAGSystem()

# --- Gemini Client Initialization ---
# Load GEMINI_API_KEY from environment variable.
# For Canvas runtime, it will be provided automatically if left empty.
# For local testing, ensure you set it as an environment variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here") # CHANGE THIS FOR PRODUCTION!
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

async def call_gemini_api(prompt: str) -> str | None:
    """
    Calls the Gemini API to generate text.
    Implements exponential backoff for retries.
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set. Cannot call Gemini API.")
        return None

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    payload = {"contents": chat_history}
    headers = {'Content-Type': 'application/json'}
    
    retries = 0
    max_retries = 5
    base_delay = 1 # seconds

    while retries < max_retries:
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Gemini API response structure unexpected: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Gemini API call failed (attempt {retries + 1}/{max_retries}): {e}")
            retries += 1
            if retries < max_retries:
                delay = base_delay * (2 ** retries)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached for Gemini API call.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding Gemini API JSON response: {e}. Response text: {response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Gemini API call: {e}")
            return None
    return None

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the HackRx RAG Policy Assistant API!"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    pdf_file: UploadFile = File(...),
    questions_json: str = File(...),
    auth: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Processes an uploaded PDF document and answers questions based on its content using RAG.
    """
    try:
        request_data = json.loads(questions_json)
        questions = request_data.get("questions", [])
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON format for questions. Ensure 'questions' is a JSON string."
        )

    print(f"Received request for PDF file '{pdf_file.filename}' with {len(questions)} questions.")

    pdf_bytes = await pdf_file.read()
    full_text = parse_pdf_from_bytes(pdf_bytes)
    
    if not full_text:
        return QueryResponse(answers=["Error: Could not process the provided PDF file. Please ensure it's a valid PDF."])

    chunks = chunk_text(full_text)
    if not chunks:
        return QueryResponse(answers=["Error: No readable text found in the document after chunking."])
    
    rag_system.create_index(chunks)
    
    answers = []
    for question in questions:
        print(f"Processing question: '{question}'")
        relevant_chunks = rag_system.search(question, k=3)
        
        if not relevant_chunks:
            answers.append("No relevant information found in the policy for this question.")
            print("No relevant chunks found.")
            continue

        context = "\n".join(relevant_chunks)
        print(f"Context for LLM: {context[:200]}...")

        prompt = f"""
        Based on the following policy text, answer the user's question concisely and directly.
        If the answer cannot be found in the provided policy text, state "The policy does not contain information to answer this question."
        
        Policy Text:
        "{context}"
        
        User's Question:
        "{question}"
        
        Answer:
        """
        
        answer = await call_gemini_api(prompt)

        if answer:
            if "not contain information" in answer.lower() or "cannot answer" in answer.lower() or "not found" in answer.lower():
                answers.append("The policy does not contain information to answer this question.")
            else:
                answers.append(answer)
            print(f"Gemini LLM Answer: {answer}")
        else:
            answers.append("Error: Could not generate an answer from Gemini API.")
            print("Gemini API call failed.")
    
    return QueryResponse(answers=answers)

# --- Uvicorn Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

