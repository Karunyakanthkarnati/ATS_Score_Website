import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
import PyPDF2

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# ==============================
# FILE EXTRACTION
# ==============================
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""
    return text

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    if Document is None:
        return ""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

def extract_text_from_image(file_path):
    """Extract text from image using OCR"""
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting from image: {e}")
        return ""

def extract_text_from_file(file_path):
    """Detect file type and extract text accordingly"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        return extract_text_from_image(file_path)
    else:
        # Try to read as text
        return extract_text_from_txt(file_path)

# ==============================
# RESUME PARSER (LLM)
# ==============================
def parse_resume(resume_text):
    prompt = f"""
You are a resume parser.

Extract:
- Skills
- Experience summary
- Education
- Tools & technologies

Resume:
{resume_text}

Return in bullet points.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error in parse_resume: {e}")
        return "Error parsing resume. Please try again."

# ==============================
# JOB DESCRIPTION PARSER
# ==============================
def parse_job_description(jd_text):
    prompt = f"""
Extract:
- Required skills
- Responsibilities
- Preferred qualifications

Job Description:
{jd_text}

Return in bullet points.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error in parse_job_description: {e}")
        return "Error parsing job description. Please try again."

# ==============================
# ATS MATCHING
# ==============================
def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
You are an Applicant Tracking System.

Compare the resume and job description.

Resume:
{parsed_resume}

Job Description:
{parsed_jd}

Provide your analysis in the following format:

### 1. Match Percentage
**XX%** (where XX is a number between 0-100)

### 2. Matching Skills
- List matching skills here

### 3. Missing Skills
- List missing skills here

### 4. Strengths
- **Strength 1**: Details about why this is a strength
- **Strength 2**: Details about why this is a strength

### 5. Improvement Suggestions
- **Suggestion 1**: Details about what needs improvement
- **Suggestion 2**: Details about what needs improvement
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error in ats_match: {e}")
        return "Error analyzing resume. Please try again."

# ==============================
# API ROUTE (PDF UPLOAD)
# ==============================

@app.route('/', methods=['GET'])

def index():
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume file is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400
    
    if not resume_file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Save file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
    resume_file.save(file_path)

    # Extract resume text from any file type
    resume_text = extract_text_from_file(file_path)

    # Parse using Gemini
    parsed_resume = parse_resume(resume_text)
    parsed_jd = parse_job_description(jd_text)

    # ATS Matching
    ats_result = ats_match(parsed_resume, parsed_jd)

    return jsonify({
        "parsed_resume": parsed_resume,
        "parsed_job_description": parsed_jd,
        "ats_result": ats_result
    })

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8080)