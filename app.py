import os
import sqlite3
import pdfplumber
import pytesseract
import cv2
import requests
import markdown
import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Configuration ---
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"
Own_API_KEY = "YOUR_API_KEY_HERE"
  # <--- NEW: Link to your local brain

# --- Tesseract Setup (Same as before) ---
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    os.path.join(os.environ.get('LOCALAPPDATA', ''), r'Tesseract-OCR\tesseract.exe')
]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Database Setup ---
def get_db_connection():
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- NEW: The Smart Brain Function ---
def ask_local_llama(prompt, context=""):
    """Sends text to your local Ollama (Llama 3) brain."""
    system_instruction = (
        "You are SESKO, a highly advanced AI assistant created by Kaif. "
        "You are helpful, witty, and precise. "
        "If the user asks about previous files, use the provided context."
    )
    
    payload = {
        "model": "llama3.2",
        "prompt": f"System: {system_instruction}\nContext: {context}\nUser: {prompt}",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "Brain error.")
        else:
            return "Error: My local brain is not responding. Is Ollama running?"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- Existing Tools (Search, OCR, etc.) ---
def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        text = soup.get_text()[:2000] # Limit text for the brain
        return f"WEBSITE CONTENT ({url}):\n{text}"
    except Exception as e:
        return f"Error reading website: {str(e)}"

def google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute()
        items = res.get('items', [])
        if not items: return "No results found."
        
        results_text = ""
        for item in items:
            results_text += f"- {item['title']}: {item['snippet']}\n"
        
        # We send the search results to the brain to summarize!
        return ask_local_llama(f"Summarize these search results for the user: {results_text}")
    except Exception as e:
        return f"Search Error: {str(e)}"

def process_files_for_context():
    """Reads the last uploaded file to give context to the brain."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT file_path FROM history WHERE file_path IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    
    if not row or not os.path.exists(row['file_path']):
        return ""

    path = row['file_path']
    try:
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "IMAGE CONTENT: " + pytesseract.image_to_string(Image.open(path)).strip()[:1000]
        else:
            with pdfplumber.open(path) as pdf:
                return "PDF CONTENT: " + " ".join([page.extract_text() or "" for page in pdf.pages])[:1500]
    except:
        return ""

# --- Main Routes ---

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_text = data.get('message', '').strip()
    mode = data.get('mode', 'chat')
    
    final_response = ""

    # 1. MODE: SEARCH (User clicked Globe Icon)
    if mode == 'search':
        final_response = google_search(user_text)

    # 2. MODE: NORMAL CHAT
    else:
        # A. If User sends a link, read it + Ask Brain to summarize
        if "http" in user_text.lower():
            urls = [w for w in user_text.split() if w.startswith('http')]
            website_content = scrape_website(urls[0])
            final_response = ask_local_llama(f"Analyze this website content: {user_text}", context=website_content)

        # B. If User asks about "this file" or "the image", check uploads
        elif any(w in user_text.lower() for w in ["file", "pdf", "image", "document", "read this"]):
            file_context = process_files_for_context()
            if file_context:
                final_response = ask_local_llama(user_text, context=file_context)
            else:
                final_response = "I don't see any recent files to read."

        # C. Normal Conversation (The Smart Brain!)
        else:
            # We just send the text directly to Llama
            final_response = ask_local_llama(user_text)

    # Convert Markdown (Bold, Italic) to HTML for the website
    return jsonify({'reply': markdown.markdown(final_response)})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"reply": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"reply": "No selected file"})
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    with get_db_connection() as conn:
        conn.execute("INSERT INTO history (user_text, bot_text, file_path) VALUES (?, 'Received', ?)", 
                     (f"Uploaded: {filename}", file_path))
        conn.commit()
    
    # Let the brain know a file arrived
    return jsonify({"reply": markdown.markdown(f"✅ **{filename}** received. You can now ask me questions about it!")})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)