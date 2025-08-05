# client.py
import requests
import json
import os
from typing import List

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/hackrx/run"
# Load API_TOKEN from environment variable.
API_TOKEN = os.getenv("API_TOKEN") 
if not API_TOKEN:
    # Fallback for local development if not set, but warn
    print("WARNING: API_TOKEN environment variable not set in client. Using a default for development.")
    API_TOKEN = "your-super-secret-development-token" # CHANGE THIS FOR PRODUCTION!

def send_query_to_api(pdf_path: str, questions: List[str]) -> List[str]:
    """
    Sends a POST request to the FastAPI RAG endpoint with an uploaded PDF and questions.
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    try:
        with open(pdf_path, "rb") as f:
            files = {'pdf_file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {'questions_json': json.dumps({"questions": questions})} # Questions as a JSON string

            print(f"\nSending PDF '{os.path.basename(pdf_path)}' and {len(questions)} question(s) to the API...")
            response = requests.post(API_URL, headers=headers, files=files, data=data)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_data = response.json()
            return response_data.get("answers", [])
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'. Please check the path.")
        return ["API Error: PDF file not found."]
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the API: {e}")
        if response and response.status_code == 401:
            print("Authentication failed. Please check your API token.")
        return [f"API Error: {e}"]
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        return ["API Error: Invalid JSON response."]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [f"An unexpected error occurred: {e}"]

def main():
    print("Welcome to the Interactive PDF Policy Assistant!")
    print("Type 'exit' to quit at any time.")

    pdf_path = input("Enter the path to your local PDF file: ").strip()
    if not pdf_path:
        print("PDF path cannot be empty. Exiting.")
        return
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'. Please provide a valid path.")
        return

    print(f"Using local PDF: {pdf_path}")

    print("\nProcessing document and building knowledge base... This might take a moment.")
    initial_response = send_query_to_api(pdf_path, ["What is this document about?"])
    if "Error" in initial_response[0] or "not contain information" in initial_response[0]:
        print(f"Initial document processing failed: {initial_response[0]}")
        print("Please ensure the FastAPI server is running correctly and the PDF is valid.")
        return
    else:
        print("Document processed successfully. You can now ask questions.")
        print(f"Initial document summary: {initial_response[0]}")


    while True:
        user_question = input("\nEnter your question: ").strip()
        if user_question.lower() == 'exit':
            print("Exiting the assistant. Goodbye!")
            break

        if not user_question:
            print("Please enter a question.")
            continue

        answers = send_query_to_api(pdf_path, [user_question])
        
        for i, answer in enumerate(answers):
            print(f"\nAnswer {i+1}: {answer}")

if __name__ == "__main__":
    main()

