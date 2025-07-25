import streamlit as st
import pdfplumber
import docx
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")

# --------- Helpers ---------
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
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

def extract_keywords(text, top_n=30):
    words = nltk.word_tokenize(clean_text(text))
    tagged = nltk.pos_tag(words)
    keywords = [
        word for word, pos in tagged
        if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ')
    ]
    return [word for word, _ in Counter(keywords).most_common(top_n)]

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

def analyze_keywords(cv_text, jd_text):
    jd_keywords = extract_keywords(jd_text)
    cv_keywords = extract_keywords(cv_text)

    missing = []
    matched = []
    synonym_matches = []

    for word in jd_keywords:
        if word in cv_keywords:
            matched.append(word)
        else:
            found = False
            for cv_word in cv_keywords:
                if word in find_synonyms(cv_word):
                    synonym_matches.append((word, cv_word))
                    found = True
                    break
            if not found:
                missing.append(word)
    return matched, missing, synonym_matches, jd_keywords

def extract_experience_sentences(cv_text):
    lines = cv_text.split("\n")
    return [line.strip() for line in lines if re.search(r"\b(led|managed|developed|designed|implemented|improved|responsible|worked|achieved)\b", line, re.IGNORECASE)][:5]

def generate_improvement_summary(missing, synonyms, example_lines):
    suggestions = []

    if missing:
        suggestions.append(f"🛠 Add keywords like: {', '.join(missing)} to better match the JD.")
    
    if synonyms:
        for jd_word, cv_word in synonyms:
            suggestions.append(f"🔁 Consider changing '{cv_word}' to '{jd_word}' for a closer match to the JD.")

    if example_lines:
        suggestions.append("📌 Based on your CV, consider improving these lines:")
        for line in example_lines:
            suggestions.append(f"→ {line}")
        suggestions.append("Can you make them more results-oriented or align with key job verbs/nouns?")

    suggestions.append("\n💡 Reflect on your Unique Strengths:")
    suggestions.append("- What’s one achievement you’re proud of?")
    suggestions.append("- How did you make a process better or faster?")
    suggestions.append("- What feedback did your manager give that stood out?")

    return "\n\n".join(suggestions)

# --------- UI ---------
st.title("📄 CV vs JD Matcher (ATS Style)")
st.write("Upload your CV and Job Description to see how well they align.")

cv_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("Upload the Job Description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("🔍 Analyze Match") and cv_file and jd_file:
    with st.spinner("Analyzing your CV and JD..."):
        cv_text = extract_text(cv_file)
        jd_text = extract_text(jd_file)

        score = calculate_match_score(cv_text, jd_text)
        matched, missing, synonyms, jd_keywords = analyze_keywords(cv_text, jd_text)
        example_lines = extract_experience_sentences(cv_text)
        suggestions = generate_improvement_summary(missing, synonyms, example_lines)

    st.success(f"✅ Match Score: {score}%")
    st.markdown("### 📊 Job Description Keywords")
    st.write(", ".join(jd_keywords))

    if missing:
        st.markdown("### ❌ Missing Keywords in CV")
        st.write(", ".join(missing))
    if synonyms:
        st.markdown("### 🔁 Synonym Suggestions")
        for jd_word, cv_word in synonyms:
            st.write(f"- Consider replacing **{cv_word}** with **{jd_word}**")

    st.markdown("### ✅ Suggestions to Improve Your CV")
    st.write(suggestions)
