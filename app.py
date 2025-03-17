import os
import tempfile
import numpy as np
import uuid
import logging
import json
from datetime import datetime
from collections import deque
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64
import docx
import openpyxl
import csv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 5120 * 1024 * 1024  # 512MB max upload size
app.config['SESSION_COOKIE_SECURE'] = True

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up ChromaDB
client = chromadb.PersistentClient("./chroma_db")
try:
    collection = client.get_collection("documents")
    logger.info("Using existing ChromaDB collection")
except:
    collection = client.create_collection("documents")
    logger.info("Created new ChromaDB collection")

# Initialize Groq client
groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

# Conversation history storage
conversation_history = {}
feedback_data = []

# Document Processing Functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a Word document."""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a plain text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}")
        return ""

def extract_text_from_excel(excel_path):
    """Extract text from an Excel file."""
    try:
        wb = openpyxl.load_workbook(excel_path, data_only=True)
        text = ""
        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                row_text = " ".join(str(cell.value) if cell.value is not None else "" for cell in row)
                if row_text.strip():
                    text += row_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Excel: {e}")
        return ""

def extract_text_from_csv(csv_path):
    """Extract text from a CSV file."""
    try:
        text = ""
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
            reader = csv.reader(file)
            for row in reader:
                text += " ".join(row) + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from CSV: {e}")
        return ""

def process_document(file_path):
    """Process different document types based on file extension."""
    extension = file_path.split('.')[-1].lower()
    
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension == 'txt':
        return extract_text_from_txt(file_path)
    elif extension in ['xlsx', 'xls']:
        return extract_text_from_excel(file_path)
    elif extension == 'csv':
        return extract_text_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def chunk_text(text, source_name):
    """Split text into manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_chunks = text_splitter.split_text(text)
        return [{"text": chunk, "source": source_name} for chunk in doc_chunks]
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return []

def get_embeddings(texts):
    """Generate embeddings for text chunks."""
    try:
        return embedding_model.encode(texts).tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

def store_chunks_in_db(chunks):
    """Store text chunks and their embeddings in ChromaDB."""
    try:
        if not chunks:
            return False
            
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)
        ids = [f"chunk_{uuid.uuid4()}" for _ in range(len(chunks))]
        metadata = [{"source": chunk["source"]} for chunk in chunks]
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        return True
    except Exception as e:
        logger.error(f"Error storing chunks in database: {e}")
        return False

def query_documents_with_tfidf(query, top_k=5):
    """Retrieve relevant document chunks using TF-IDF and cosine similarity."""
    try:
        # Get all documents from the collection
        all_docs = collection.get()
        if not all_docs["documents"]:
            return [], []
            
        # Create corpus with query and documents
        corpus = [query] + all_docs["documents"]
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()
        
        # Calculate similarity
        cosine_matrix = cosine_similarity(vectors)
        similarity_scores = cosine_matrix[0][1:]  # Exclude query itself
        
        # Get top-k indices
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]  # Reverse to get highest first
        
        contexts = [all_docs["documents"][i] for i in top_indices]
        sources = [all_docs["metadatas"][i]["source"] for i in top_indices]
        
        return contexts, sources
    except Exception as e:
        logger.error(f"Error querying documents with TF-IDF: {e}")
        return [], []

def query_documents_with_embeddings(query, top_k=5):
    """Retrieve relevant document chunks using embedding similarity."""
    try:
        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results["documents"][0]:
            return [], []
            
        contexts = results["documents"][0]
        sources = [meta["source"] for meta in results["metadatas"][0]]
        
        return contexts, sources
    except Exception as e:
        logger.error(f"Error querying documents with embeddings: {e}")
        return [], []

def hybrid_query_documents(query, top_k=5):
    """Hybrid retrieval using both TF-IDF and embeddings."""
    try:
        # Get results from both methods
        tfidf_contexts, tfidf_sources = query_documents_with_tfidf(query, top_k)
        embedding_contexts, embedding_sources = query_documents_with_embeddings(query, top_k)
        
        # Combine results, removing duplicates
        seen = set()
        combined_contexts = []
        combined_sources = []
        
        # Add TF-IDF results first (usually more precise)
        for context, source in zip(tfidf_contexts, tfidf_sources):
            if context not in seen:
                seen.add(context)
                combined_contexts.append(context)
                combined_sources.append(source)
        
        # Add embedding results
        for context, source in zip(embedding_contexts, embedding_sources):
            if context not in seen and len(combined_contexts) < top_k:
                seen.add(context)
                combined_contexts.append(context)
                combined_sources.append(source)
        
        return combined_contexts, combined_sources
    except Exception as e:
        logger.error(f"Error in hybrid document query: {e}")
        return [], []

def generate_response(query, contexts, session_id):
    """Generate a response using Groq API based on retrieved contexts and conversation history."""
    try:
        # Get conversation history for this session
        history = conversation_history.get(session_id, deque(maxlen=10))
        
        # Combine contexts into a single context string
        context_text = "\n\n".join(contexts)
        
        # Create conversation history string
        history_text = ""
        if history:
            history_text = "Previous conversation:\n"
            for i, (q, a) in enumerate(history):
                history_text += f"User: {q}\nAssistant: {a}\n\n"
        
        # Create prompt for Groq with improved instructions
        prompt = f"""You are an assistant that answers questions about products and prices based on the provided information.
        
{history_text}

Context from documents:
{context_text}

Current Question: {query}

Important instructions:
1. Provide direct answers without phrases like "Based on the provided context" or "According to the information"
2. Always provide unit prices unless the user specifically asks about multiple units
3. ALWAYS share full product description with prices in TABLE MUST
4. If no exact matching, share similar items
5. Be concise and focus on the specific information requested
6. If the answer cannot be found in the context, say "I don't have this information"

Please answer the question based on the provided information.
"""
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            model= "gemma2-9b-it", # "gemma2-9b-it", # "mixtral-8x7b-32768", # "llama3-70b-8192",  #llama-3.3-70b-versatile #mixtral-8x7b-32768 #qwen-2.5-coder-32b
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        # Update conversation history
        history.append((query, answer))
        conversation_history[session_id] = history
        
        return answer
    except Exception as e:
        logger.error(f"Error generating response with Groq: {e}")
        return "Sorry, I encountered an error while generating a response."

def rag_bot(query, session_id):
    """Main RAG function to process a query and return an answer."""
    contexts, sources = hybrid_query_documents(query)
    if not contexts:
        return {
            "answer": "I don't have any relevant information to answer this question. Try uploading some documents first.",
            "sources": []
        }
    
    response = generate_response(query, contexts, session_id)
    return {
        "answer": response,
        "sources": sources
    }

def generate_document_stats():
    """Generate statistics about the documents in the database."""
    try:
        all_docs = collection.get()
        if not all_docs["documents"]:
            return None
            
        # Count documents per source
        source_counts = {}
        for meta in all_docs["metadatas"]:
            source = meta["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Generate plot using Figure directly (thread-safe approach)
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(list(source_counts.keys()), list(source_counts.values()), color='skyblue')
        ax.set_xlabel('Document Source')
        ax.set_ylabel('Number of Chunks')
        ax.set_title('Document Chunks by Source')
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buffer)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graph = base64.b64encode(image_png).decode('utf-8')
        return graph
    except Exception as e:
        logger.error(f"Error generating document stats: {e}")
        return None

def get_file_extension_stats():
    """Generate statistics about document types in the database."""
    try:
        all_docs = collection.get()
        if not all_docs["documents"]:
            return None
            
        # Count documents per extension
        extension_counts = {}
        for meta in all_docs["metadatas"]:
            source = meta["source"]
            extension = source.split('.')[-1].lower() if '.' in source else 'unknown'
            extension_counts[extension] = extension_counts.get(extension, 0) + 1
        
        # Generate plot using Figure directly (thread-safe approach)
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(list(extension_counts.keys()), list(extension_counts.values()), color='lightgreen')
        ax.set_xlabel('File Type')
        ax.set_ylabel('Number of Chunks')
        ax.set_title('Document Chunks by File Type')
        fig.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buffer)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graph = base64.b64encode(image_png).decode('utf-8')
        return graph
    except Exception as e:
        logger.error(f"Error generating file extension stats: {e}")
        return None

# Flask Routes
@app.route('/')
def index():
    """Render the main page."""
    # Set a session cookie if not present
    resp = make_response(render_template('index.html'))
    if not request.cookies.get('session_id'):
        session_id = str(uuid.uuid4())
        resp.set_cookie('session_id', session_id, httponly=True)
    return resp

@app.route('/admin/upload')
def upload_page():
    """Render the upload/admin page."""
    resp = make_response(render_template('upload.html'))
    if not request.cookies.get('session_id'):
        session_id = str(uuid.uuid4())
        resp.set_cookie('session_id', session_id, httponly=True)
    return resp

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document file uploads."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get file extension
    filename = secure_filename(file.filename)
    extension = filename.split('.')[-1].lower() if '.' in filename else ''
    supported_extensions = ['pdf', 'docx', 'txt', 'xlsx', 'xls', 'csv']
    
    if extension not in supported_extensions:
        return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"}), 400
    
    try:
        # Save to temporary file to avoid memory issues with large files
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Process the document
        text = process_document(temp_path)
        if not text:
            os.unlink(temp_path)  # Clean up temp file
            return jsonify({"error": f"Could not extract text from {filename}"}), 400
            
        chunks = chunk_text(text, filename)
        success = store_chunks_in_db(chunks)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if success:
            return jsonify({
                "message": f"Successfully processed {filename}", 
                "chunks": len(chunks),
                "file_type": extension
            })
        else:
            return jsonify({"error": "Failed to store document in database"}), 500
            
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query_text = data['query']
    session_id = request.cookies.get('session_id', str(uuid.uuid4()))
    
    # Generate a unique ID for this query
    query_id = str(uuid.uuid4())
    
    result = rag_bot(query_text, session_id)
    result['query_id'] = query_id
    
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback on responses."""
    data = request.json
    if not data or 'rating' not in data or 'query_id' not in data:
        return jsonify({"error": "Missing required feedback data"}), 400
    
    try:
        rating = int(data['rating'])
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be between 1 and 5"}), 400
            
        query_id = data['query_id']
        session_id = request.cookies.get('session_id', 'unknown')
        
        # Store feedback
        feedback_entry = {
            "query_id": query_id,
            "session_id": session_id,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
            "comment": data.get('comment', '')
        }
        
        feedback_data.append(feedback_entry)
        
        # Save feedback to file
        with open('feedback.json', 'w') as f:
            json.dump(feedback_data, f)
        
        return jsonify({"message": "Feedback received, thank you!"})
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Generate and return document statistics."""
    try:
        graph = generate_document_stats()
        file_type_graph = get_file_extension_stats()
        
        if graph and file_type_graph:
            return jsonify({
                "graph": graph,
                "file_type_graph": file_type_graph
            })
        else:
            return jsonify({"error": "No documents available or error generating stats"}), 404
    except Exception as e:
        logger.error(f"Error in stats route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for the current session."""
    session_id = request.cookies.get('session_id')
    if session_id and session_id in conversation_history:
        conversation_history[session_id] = deque(maxlen=10)
        return jsonify({"message": "Conversation history cleared"})
    return jsonify({"message": "No history to clear"})

@app.route('/supported_formats', methods=['GET'])
def supported_formats():
    """Return list of supported file formats."""
    formats = {
        'pdf': 'PDF Documents',
        'docx': 'Word Documents',
        'txt': 'Text Files',
        'xlsx': 'Excel Spreadsheets',
        'xls': 'Excel Spreadsheets (older format)',
        'csv': 'CSV Files'
    }
    return jsonify(formats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)