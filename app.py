import streamlit as st
import pdfplumber
import docx
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")

def extract_text(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode('utf-8')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def get_top_keywords(text, n=20):
    text = clean_text(text)
    words = text.split()
    counts = Counter(words)
    common = counts.most_common(n)
    return set([word for word, _ in common])

def suggest_improvements(cv_text, jd_text):
    cv_keywords = get_top_keywords(cv_text, n=30)
    jd_keywords = get_top_keywords(jd_text, n=30)
    missing = jd_keywords - cv_keywords
    if not missing:
        return "✅ Your CV aligns very well with the Job Description!"
    return "You may consider adding the following keywords to your CV:\n\n" + ", ".join(list(missing))

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

        st.success(f"✅ Match Score: {score}%")
        st.markdown("### ✍️ Suggestions to Improve:")
        st.write(suggestions)
    except Exception as e:
        st.error(f"❌ Error: {e}")
