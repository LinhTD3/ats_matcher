import streamlit as st
import pdfplumber
import docx
import re
import difflib
import requests
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="CV vs JD Matcher", layout="centered")
st.title("📄 CV vs JD Matcher (ATS + HR Perspective)")
st.markdown("Upload your CV and Job Description to get AI-powered matching & improvement suggestions.")

# ---------- Helper Functions ----------

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

def extract_relevant_keywords(text, top_n=30):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc 
                if token.pos_ in {"NOUN", "VERB", "ADJ"} and not token.is_stop and len(token) > 2]
    counts = Counter(keywords)
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

def grammar_check(text, max_issues=5):
    url = f"https://services.gingersoftware.com/Ginger/correct/jsonSecured/GingerTheText"
    params = {
        "lang": "US",
        "clientVersion": "2.0",
        "apiKey": "6ae0c3a0-afdc-4532-a810-82ded0054236",
        "text": text[:600]
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        suggestions = []
        for correction in data.get("Corrections", [])[:max_issues]:
            original = correction["Text"]
            suggestion = correction["Suggestions"][0]["Text"]
            suggestions.append(f"✏️ `{original}` → `{suggestion}`")
        return suggestions
    except Exception as e:
        return [f"Grammar check failed: {e}"]

def calculate_match_score(cv_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def extract_experience_sentences(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if re.search(r"\b(experience|responsible|led|managed|project|developed|created|achieved|supported|worked|improved)\b", line.lower()) and len(line.strip().split()) >= 5]

def generate_open_questions(sentences, jd_keywords):
    prompts = []
    for s in sentences[:3]:  # Limit to 3 for clarity
        suggestions = []
        if "responsible for" in s.lower():
            suggestions.append("Can you replace 'responsible for' with a strong action verb (e.g., led, executed, launched)?")
        if not re.search(r"\d+%|\$\d+|\d+ users|\d+ projects|\d+ clients", s):
            suggestions.append("Can you add a metric to show scale or impact? For example, how many users, clients, or revenue?")
        for kw in jd_keywords:
            if kw in s.lower():
                suggestions.append(f"Does this sentence clearly show how you used '{kw}'? Could you make it more specific?")
        if suggestions:
            prompts.append(f"🧠 **From your CV:**\n> {s}\n" + "\n".join(["- " + q for q in suggestions]))
    return prompts

def generate_summary(score, missing, similar, grammar_issues, cv_sentences, jd_keywords):
    summary = f"### ✅ Match Score: **{score}%**\n\n"
    summary += "Great job!\n\n" if score >= 80 else \
               "You’re on the right path. Let’s fine-tune your CV.\n\n" if score >= 60 else \
               "There's a gap to close – here's how to improve.\n\n"

    if missing:
        summary += f"**📌 Missing Keywords from JD:**\n- {', '.join(list(missing.keys())[:8])}\n\n"
    if similar:
        summary += f"**🔁 Consider Rephrasing These Terms for Better Alignment:**\n"
        summary += "\n".join([f"`{v}` → `{k}`" for k, v in similar.items()]) + "\n\n"
    if grammar_issues:
        summary += f"**📝 Grammar Tips:**\n" + "\n".join(grammar_issues) + "\n\n"

    summary += "---\n### 💡 Personalized CV Enhancement Tips:\n"
    insights = generate_open_questions(cv_sentences, jd_keywords)
    if insights:
        for tip in insights:
            summary += tip + "\n\n"
    else:
        summary += "- Try using stronger action verbs.\n"
        summary += "- Use quantifiable results (e.g. Increased sales by 20%).\n"
        summary += "- Align your experience directly with job description terms.\n"

    summary += "---\n### 🔍 Self-Reflection Questions to Define Your USP:\n"
    summary += "- What impact have you made in previous roles?\n"
    summary += "- What skills do you repeatedly use that make you stand out?\n"
    summary += "- Can your experience solve this employer's current problems?\n"

    return summary

# ---------- UI ----------

cv_file = st.file_uploader("📄 Upload your CV (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
jd_file = st.file_uploader("📝 Upload the Job Description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("🔍 Analyze Match") and cv_file and jd_file:
    with st.spinner("Analyzing content, extracting keywords, checking grammar..."):
        cv_text = extract_text(cv_file)
        jd_text = extract_text(jd_file)

        if not cv_text.strip() or not jd_text.strip():
            st.error("❌ One of your files is empty or unreadable.")
        else:
            cv_keywords = extract_relevant_keywords(cv_text)
            jd_keywords = extract_relevant_keywords(jd_text)
            missing = find_missing_keywords(jd_keywords, cv_keywords)
            similar = find_similar_keywords(jd_keywords, cv_keywords)
            grammar_issues = grammar_check(cv_text)
            score = calculate_match_score(cv_text, jd_text)
            cv_sentences = extract_experience_sentences(cv_text)

            st.markdown("### 🧠 JD Keywords (Most Frequent & Relevant)")
            st.json(jd_keywords)

            st.markdown("### ❌ Keywords Missing in Your CV")
            st.write(", ".join(missing.keys()) if missing else "All key terms found in your CV!")

            st.markdown("### 🔁 Similar but Not Exact Terms")
            if similar:
                for k, v in similar.items():
                    st.write(f"`{v}` → `{k}`")
            else:
                st.write("✅ Great! No suggestions for keyword replacement.")

            st.markdown("### ✍️ Grammar & Spelling Suggestions")
            if grammar_issues:
                for issue in grammar_issues:
                    st.write(issue)
            else:
                st.write("✅ No major grammar issues detected.")

            st.markdown("### 📌 Summary & Personalized Advice")
            st.markdown(generate_summary(score, missing, similar, grammar_issues, cv_sentences, jd_keywords))
