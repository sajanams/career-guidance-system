from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import PyPDF2
from docx import Document

app = Flask(__name__)

# Load the CSV data
jobs = pd.read_csv("career_recommendation_dataset.csv")  # Ensure the CSV file is in the same directory as app.py

# Pre-trained TF-IDF vectorizer
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(jobs["skills"])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for skill-based job recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    skills = data.get("skills", "")

    # Vectorize user skills
    user_vector = vectorizer.transform([skills])

    # Calculate cosine similarity between user skills and job descriptions
    similarities = cosine_similarity(user_vector, job_vectors).flatten()

    # Get top 5 job recommendations
    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = jobs.iloc[top_indices].to_dict(orient="records")

    return jsonify(recommendations)

# Route for resume-based job recommendations
@app.route("/upload", methods=["POST"])
def upload():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    filename = file.filename

    # Extract text based on file type
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Vectorize resume text
    user_vector = vectorizer.transform([text])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, job_vectors).flatten()

    # Get top 5 job recommendations
    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = jobs.iloc[top_indices].to_dict(orient="records")

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)