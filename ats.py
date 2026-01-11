import os, json, re
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
import PyPDF2

try:
    from docx import Document
except ImportError:
    Document = None

# ================= INIT =================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= FILE EXTRACTION =================

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text += p.extract_text() or ""
    return text

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_docx(path):
    if not Document:
        return ""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return extract_text_from_pdf(path)
    if ext == ".txt": return extract_text_from_txt(path)
    if ext == ".docx": return extract_text_from_docx(path)
    return ""

# ================= JSON CLEANER =================

def clean_json(text):
    text = text.strip()
    text = re.sub(r"```json|```", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end]

# ================= LLM =================

def ats_match(resume, jd):
    prompt = f"""
You are an advanced ATS.

Return ONLY valid JSON in this format:

{{
 "match_score": number,
 "summary": string,
 "strengths": [string],
 "missing_skills": [string],
 "improvement_suggestions": [string],
 "resume_breakdown": string
}}

Resume:
{resume}

Job Description:
{jd}
"""
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return res.text.strip()

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    jd_text = request.form.get("job_description")

    if not resume_file or not jd_text:
        return jsonify({"error": "Resume and Job Description required"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
    resume_file.save(path)

    resume_text = extract_text(path)
    raw_json = ats_match(resume_text, jd_text)
    cleaned = clean_json(raw_json)

    try:
        ats_data = json.loads(cleaned)
        return jsonify(ats_data)

    except Exception:
        print("RAW AI OUTPUT:\n", raw_json)
        print("CLEANED JSON:\n", cleaned)
        return jsonify({"error": "Invalid AI JSON format"}), 500

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True, port=8080)
