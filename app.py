import streamlit as st
import pdfplumber
import docx
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


st.set_page_config(page_title="CV vs JD Matcher", layout="centered")

import spacy
nlp = spacy.load("en_core_web_sm")

def extract_text(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode('utf-8')

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def extract_skills(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 2]

def suggest_improvements(cv_text, jd_text):
    cv_skills = set(extract_skills(cv_text))
    jd_skills = set(extract_skills(jd_text))
    missing = jd_skills - cv_skills
    if not missing:
        return "Your CV aligns well with the JD!"
    return "You may consider including the following keywords:\n\n" + ", ".join(list(missing)[:10])

# UI
st.title("📄 CV vs JD Matcher (ATS Style)")
st.write("Upload your CV and Job Description to see how well they match.")

cv_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("Upload the Job Description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("🔍 Analyze Match") and cv_file and jd_file:
    try:
        with st.spinner("Extracting and analyzing..."):
            cv_text = extract_text(cv_file)
            jd_text = extract_text(jd_file)

            score = calculate_match_score(cv_text, jd_text)
            suggestions = suggest_improvements(cv_text, jd_text)

        st.success(f"Match Score: {score}%")
        st.markdown("### ✍️ Suggestions to Improve:")
        st.write(suggestions)
    except Exception as e:
        st.error(f"An error occurred: {e}")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

