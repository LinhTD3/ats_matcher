import streamlit as st
import pdfplumber
import docx
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from textblob import TextBlob
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")
st.title("📄 CV vs JD Matcher (ATS Style)")
st.markdown("Upload your CV and Job Description to receive smart feedback and improvement suggestions.")

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
    else:
        return ""

def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)

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

def grammar_check(text):
    blob = TextBlob(text)
    corrections = []
    for sentence in blob.sentences[:5]:
        corrected = sentence.correct()
        if corrected != sentence:
            corrections.append(f"✏️ '{sentence}' → '{corrected}'")
    return corrections

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def generate_summary(score, missing, similar, grammar_issues):
    summary = f"Your match score is **{score}%**.\n\n"
    if score > 80:
        summary += "🎉 Great alignment overall!\n\n"
    elif score > 50:
        summary += "👍 You're on the right track, but a few tweaks can help.\n\n"
    else:
        summary += "⚠️ Your CV and JD need stronger alignment.\n\n"

    if missing:
        summary += f"🔍 Missing keywords: `{', '.join(list(missing.keys())[:5])}`\n\n"
    if similar:
        examples = [f"`{v}` → consider changing to `{k}`" for k, v in similar.items()]
        summary += "🪄 Similar terms you could align:\n" + "\n".join(examples[:5]) + "\n\n"
    if grammar_issues:
        summary += "📝 Spelling/Grammar Suggestions:\n" + "\n".join(grammar_issues) + "\n\n"
    return summary

def ask_usp_questions():
    st.markdown("### 💡 Define Your Unique Strength")
    st.markdown("- 🧠 What result or achievement are you most proud of?")
    st.markdown("- 🚀 What do you do better than most people in your field?")
    st.markdown("- 🎯 What’s one reason a recruiter should remember you?")

# ---------- Streamlit UI ----------

cv_file = st.file_uploader("📄 Upload your CV (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
jd_file = st.file_uploader("📝 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if st.button("🔍 Analyze Match") and cv_file and jd_file:
    with st.spinner("Analyzing your documents..."):
        cv_text = extract_text(cv_file)
        jd_text = extract_text(jd_file)

        if not cv_text.strip() or not jd_text.strip():
            st.error("❌ One of your files is empty or unreadable.")
        else:
            cv_keywords = extract_keywords(cv_text)
            jd_keywords = extract_keywords(jd_text)

            missing_keywords = find_missing_keywords(jd_keywords, cv_keywords)
            similar_keywords = find_similar_keywords(jd_keywords, cv_keywords)
            grammar_issues = grammar_check(cv_text)
            match_score = calculate_match_score(cv_text, jd_text)

            # Results
            st.success(f"✅ Match Score: {match_score}%")

            st.markdown("### 📊 Top JD Keywords")
            st.json(jd_keywords)

            st.markdown("### ❌ Missing in CV")
            st.write(", ".join(missing_keywords.keys()) or "All covered — nice work!")

            st.markdown("### 🔁 Similar Words Detected")
            for k, v in similar_keywords.items():
                st.write(f"• Replace `{v}` → `{k}`")

            st.markdown("### 📝 Grammar & Spelling Fixes")
            if grammar_issues:
                for issue in grammar_issues:
                    st.write(issue)
            else:
                st.write("Looks good!")

            st.markdown("### 🧠 Summary")
            st.markdown(generate_summary(match_score, missing_keywords, similar_keywords, grammar_issues))

            ask_usp_questions()
