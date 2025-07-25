import streamlit as st
import spacy
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


nltk.download("punkt")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
FILLER_WORDS = STOPWORDS.union({"we", "our", "i", "you", "the", "a", "an", "to", "of", "in", "on", "and", "for", "with", "as"})

# Load NLP model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Preprocess and tokenize
def extract_meaningful_keywords(text):
    doc = nlp(text)
    return list(set([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "VERB", "ADJ"} and token.lemma_.lower() not in FILLER_WORDS
    ]))

# Keyword matching logic
def analyze_keywords(cv_text, jd_text):
    cv_keywords = set(extract_meaningful_keywords(cv_text))
    jd_keywords = set(extract_meaningful_keywords(jd_text))

    matched = jd_keywords.intersection(cv_keywords)
    missing = jd_keywords - cv_keywords
    return matched, missing, jd_keywords

# Provide feedback on sentences
def generate_cv_feedback(cv_text, missing_keywords):
    feedback_sentences = []
    improved_sentences = []

    for sent in sent_tokenize(cv_text):
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in missing_keywords):
            continue  # already covered
        if len(sent.split()) < 5:
            continue  # likely not job experience related

        doc = nlp(sent)
        if any(token.pos_ in {"VERB", "NOUN"} for token in doc):
            suggestion = f"Consider expanding on this: \"{sent.strip()}\". Can you quantify your impact, add tools, or describe achievements?"
            feedback_sentences.append(suggestion)
            improved_sentences.append(f"{sent.strip()} ➜ Consider expanding or adding metrics.")

    return feedback_sentences, improved_sentences

# Highlight improvements in CV
def highlight_cv(cv_text, improved_sentences):
    highlighted = cv_text
    for improvement in improved_sentences:
        original = improvement.split("➜")[0].strip()
        if original in highlighted:
            highlighted = highlighted.replace(original, f"**🟩 {original}**")
    return highlighted

# Streamlit App
st.set_page_config("CV vs JD Matcher", layout="wide")
st.title("📄 CV vs Job Description Matcher")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("Upload your CV (.txt)", type=["txt"])
    if cv_file:
        cv_text = cv_file.read().decode("utf-8")

with col2:
    jd_input = st.text_area("Paste the Job Description")

if cv_file and jd_input:
    matched, missing, jd_keywords = analyze_keywords(cv_text, jd_input)
    feedback, improved_sentences = generate_cv_feedback(cv_text, missing)
    highlighted_cv = highlight_cv(cv_text, improved_sentences)

    st.subheader("🔍 Keyword Matching Summary")
    st.markdown(f"**Matched Keywords:** `{', '.join(sorted(matched))}`")
    st.markdown(f"**Missing Keywords:** `{', '.join(sorted(missing))}`")

    with st.expander("🛠️ Suggestions to Improve Your CV"):
        for tip in feedback:
            st.markdown(f"- {tip}")

    st.subheader("📝 CV Comparison View")
    view = st.radio("Select View:", ["Raw CV", "Feedback-Enhanced CV"])
    st.text_area("Your CV", value=cv_text if view == "Raw CV" else highlighted_cv, height=400)

