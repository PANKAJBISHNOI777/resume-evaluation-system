import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Evaluator")

job_description = st.text_area("Enter Job Description")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])

    score = round(similarity[0][0] * 100, 2)

    st.write("Resume Score:", score)

    if score < 50:
        st.write("Needs improvement")
    elif score < 75:
        st.write("Good resume but can improve")
    else:
        st.write("Excellent resume!")