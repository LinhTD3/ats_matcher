import streamlit as st
import pdfplumber
import docx
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import language_tool_python

# --- Streamlit config ---
st.set_page_config(page_title="CV vs JD Matcher", layout="centered")
st.title("📄 CV vs JD Matcher (ATS Style)")
st.markdown("Upload your CV and Job Description to get smart, actionable feedback.")

# ---------- Helpers ----------

def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())

def extract_keywords(text, top_n=30):
    words = clean_text(text).split()
    counts = Counter(words)
    return dict(counts.most_common(top_n))

def find_missing_keywords(jd_keywords, cv_keywords):
    return {kw: freq for kw, freq in jd_keywords.items() if kw not in cv_keywords}

def find_similar_keywords(jd_keywords, cv_keywords):
    similar = {}
    for jd_kw in jd_keywords:
        for cv_kw in cv_keywords:
            sim = difflib.SequenceMatcher(None, jd_kw, cv_kw).ratio()
            if sim > 0.75 and jd_kw != cv_kw:
                similar[jd_kw] = cv_kw
    return similar

def grammar_check(text, limit=5):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    issues = []
    for match in matches[:limit]:
        issues.append(f"✏️ {match.context.strip()} → {match.message}")
    return issues

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def generate_summary(score, missing, similar, grammar_issues):
    summary = f"**Match Score:** {score}%\n\n"
    if score >= 80:
        summary += "✅ Strong alignment with the job description.\n\n"
    elif score >= 50:
        summary += "⚠️ Fair match — some adjustments recommended.\n\n"
    else:
        summary += "🚫 Low match — your CV needs significant improvements.\n\n"

    if missing:
        summary += f"**Missing keywords:** {', '.join(list(missing.keys())[:5])}\n\n"
    if similar:
        summary += "**Similar terms detected (consider aligning):**\n"
        summary += "\n".join([f"• `{v}` → `{k}`" for k, v in similar.items()][:5]) + "\n\n"
    if grammar_issues:
        summary += "**Grammar Suggestions:**\n"
        summary += "\n".join(grammar_issues[:3])_
