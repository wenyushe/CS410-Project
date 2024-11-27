from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

from summarize import summarize_text

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from different file types
def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ''
    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == 'pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
        elif ext == 'docx':
            import docx
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            flash('No file provided.')
            return redirect(request.url)
        file = request.files['file']
        # Check if filename is valid
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        # Check file type
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the file
            text = extract_text(file_path)
            if not text.strip():
                flash('Failed to extract text from the document.')
                return redirect(request.url)

            # Summarize the text
            summary = summarize_text(text)
            return render_template('summary.html', summary=summary)
        else:
            flash('Invalid file type. Allowed types are txt, pdf, docx.')
            return redirect(request.url)
    return render_template('upload.html')
    
if __name__ == '__main__':
    app.run(debug=True)