import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Resume Screening", layout="centered")
st.title("🧠 AI-Based Resume Screening System")
st.write("Paste resume text to predict job role and match percentage.")

# -----------------------------
# Load dataset safely
# -----------------------------
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "dataset", "resumes.csv")

data = pd.read_csv(csv_path)

# 🔐 NORMALIZE COLUMN NAMES
data.columns = [col.strip().lower() for col in data.columns]

# 🔎 HANDLE ALL POSSIBLE CASES
if "resume" in data.columns and "category" in data.columns:
    X = data["resume"]
    y = data["category"]
else:
    # fallback if headers are missing or broken
    data = pd.read_csv(csv_path, header=None)
    data = data.iloc[:, :2]
    data.columns = ["resume", "category"]
    X = data["resume"]
    y = data["category"]

# -----------------------------
# Vectorization & Model
# -----------------------------
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# -----------------------------
# User input
# -----------------------------
user_resume = st.text_area(
    "📄 Resume Text",
    height=200,
    placeholder="Paste resume content here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    if user_resume.strip() == "":
        st.warning("Please paste resume text.")
    else:
        user_vector = vectorizer.transform([user_resume])
        predicted_role = model.predict(user_vector)[0]
        match_percentage = cosine_similarity(
            user_vector, X_vectorized
        ).max() * 100

        st.success(f"✅ Predicted Job Role: **{predicted_role}**")
        st.info(f"📊 Job Match Percentage: **{match_percentage:.2f}%**")
