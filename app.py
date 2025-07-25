import streamlit as st
import pdfplumber
import docx
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")

# --------- Utilities ---------
def extract_text(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode('utf-8')

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def extract_keywords(text, top_n=30):
    words = simple_tokenize(text)
    tagged = [(word, guess_pos(word)) for word in words]
    keywords = [word for word, pos in tagged if pos in {"NN", "VB", "JJ"}]
    return [word for word, _ in Counter(keywords).most_common(top_n)]

def guess_pos(word):
    if word.endswith("ing") or word.endswith("ed"):
        return "VB"
    elif word.endswith("ion") or word.endswith("ment") or word.endswith("ity"):
        return "NN"
    elif word.endswith("ive") or word.endswith("ous") or word.endswith("ful"):
        return "JJ"
    else:
        return "NN"

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def analyze_keywords(cv_text, jd_text):
    jd_keywords = extract_keywords(jd_text)
    cv_keywords = extract_keywords(cv_text)

    missing = [kw for kw in jd_keywords if kw not in cv_keywords]
    matched = [kw for kw in jd_keywords if kw in cv_keywords]
    synonyms = [(kw, cv_kw) for kw in missing for cv_kw in cv_keywords if kw in cv_kw or cv_kw in kw]
    matched_synonyms = [s[0] for s in synonyms]
    final_missing = [kw for kw in missing if kw not in matched_synonyms]

    return matched, final_missing, synonyms, jd_keywords

def extract_experience_sentences(cv_text):
    lines = cv_text.split("\n")
    return [line.strip() for line in lines if re.search(r"\b(led|managed|developed|designed|implemented|improved|responsible|worked|achieved)\b", line, re.IGNORECASE)][:5]

def generate_improvement_summary(missing, synonyms, example_lines):
    suggestions = []

    if missing:
        suggestions.append(f"🛠 Add these missing keywords to your CV: {', '.join(missing)}")
    
    if synonyms:
        for jd_kw, cv_kw in synonyms:
            suggestions.append(f"🔁 Replace or rephrase '{cv_kw}' with '{jd_kw}' to match JD terms better.")

    if example_lines:
        suggestions.append("📌 Sample experience lines to improve:")
        for line in example_lines:
            suggestions.append(f"→ {line}")
        suggestions.append("Tip: Add specific outcomes, numbers, or power verbs that reflect JD needs.")

    suggestions.append("\n💡 Reflect on your unique strengths:")
    suggestions.append("- What's a moment you made a clear impact?")
    suggestions.append("- What skills do colleagues say you're best at?")
    suggestions.append("- If you had 1 minute to impress a hiring manager, what would you say?")

    return "\n\n".join(suggestions)

# --------- UI ---------
st.title("📄 CV vs JD Matcher (ATS Style)")
st.write("Upload your CV and Job Description to see how well they align and what to improve.")

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
    st.markdown("### 📊 JD Keywords (Important Terms)")
    st.write(", ".join(jd_keywords))

    if missing:
        st.markdown("### ❌ Missing Keywords")
        st.write(", ".join(missing))

    if synonyms:
        st.markdown("### 🔁 Possible Improvements via Synonyms")
        for jd_kw, cv_kw in synonyms:
            st.write(f"- Replace **{cv_kw}** with **{jd_kw}**")

    st.markdown("### ✅ Suggestions to Improve Your CV")
    st.write(suggestions)
