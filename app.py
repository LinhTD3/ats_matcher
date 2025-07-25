import streamlit as st
import pdfplumber
import docx
import re

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy


# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")

def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8")

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ")]
    return Counter(keywords)

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def analyze_cv_sentences(cv_text, jd_keywords):
    suggestions = []
    sentences = [sent.string.strip() for sent in nlp(cv_text).sents if sent.string.strip()]
    for sent in sentences:
        tokens = [token.lemma_ for token in nlp(sent) if token.pos_ in ("NOUN", "VERB", "ADJ")]
        missing = [word for word in jd_keywords if word not in tokens]
        score = len(set(tokens) & set(jd_keywords)) / (len(set(jd_keywords)) + 1e-5)
        feedback = ""
        if score < 0.3:
            feedback += f"🔍 Missing key JD terms: {', '.join(missing[:5])}. "
            feedback += f"💡 Consider rewriting: e.g., '{rewrite_sentence(sent, jd_keywords)}'"
        suggestions.append((sent, feedback if feedback else "✅ Looks good"))
    return suggestions

def rewrite_sentence(sentence, jd_keywords):
    # This is just a placeholder for smarter rewriting using LLM
    tokens = [token.text for token in nlp(sentence)]
    enriched = sentence
    for key in jd_keywords:
        if key not in sentence.lower():
            enriched += f", with focus on {key}"
            break
    return enriched

# --- UI ---
st.title("📄 CV vs JD Matcher (ATS Style)")
st.write("Upload your CV and Job Description to analyze alignment and get improvement suggestions.")

cv_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("Upload the Job Description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("🔍 Analyze Match") and cv_file and jd_file:
    with st.spinner("Analyzing..."):
        cv_text_raw = extract_text(cv_file)
        jd_text_raw = extract_text(jd_file)

        cv_text = clean_text(cv_text_raw)
        jd_text = clean_text(jd_text_raw)

        score = calculate_match_score(cv_text, jd_text)

        jd_keyword_counter = extract_keywords(jd_text)
        jd_keywords = list(jd_keyword_counter.keys())
        jd_keywords_set = set(jd_keywords)

        sentence_feedback = analyze_cv_sentences(cv_text_raw, jd_keywords_set)

    st.success(f"✅ Match Score: {score}%")

    view_mode = st.radio("Choose CV View:", ["🔍 Enhanced Feedback View", "📄 Raw CV View"])

    if view_mode == "🔍 Enhanced Feedback View":
        st.markdown("### ✍️ CV Sentences with Feedback")
        for sent, feedback in sentence_feedback:
            st.markdown(f"- **{sent}**\n  - {feedback}")
    else:
        st.markdown("### 📄 Raw CV Content")
        st.text(cv_text_raw)

    # Summary
    st.markdown("### 📌 Summary of Suggestions")
    st.write("To improve your CV further, consider the following:")
    st.markdown(f"""
    - Use strong **action verbs** and **specific skills** found in the JD (e.g., `strategize`, `optimize`, `stakeholder`, `data-driven`).
    - Avoid vague statements like *"worked on many things"*. Instead, quantify and link to JD needs.
    - Customize terminology to match the JD keywords exactly (e.g., if JD says `stakeholders`, use the same instead of `clients`).
    """)

    # Open-ended questions to help define USP
    st.markdown("### 🤔 Define Your Unique Selling Point (USP)")
    st.write("Consider answering the following to further improve your CV:")
    st.markdown("""
    - What’s one achievement you’re most proud of? Can it be quantified?
    - What technology or method do you use that makes your work more effective than others?
    - Have you led or influenced change in a team, project, or organization?
    - Can you tailor any accomplishment to reflect the language or needs in this JD?
    """)
