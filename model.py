import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("dataset/resumes.csv")

X = data["Resume"]
y = data["Category"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

print("Model trained successfully!")

# User input
print("\nEnter Resume Text:")
user_resume = input()

user_vector = vectorizer.transform([user_resume])

# Prediction
predicted_role = model.predict(user_vector)[0]

# Match percentage
similarity_scores = cosine_similarity(user_vector, X_vectorized)
match_percentage = similarity_scores.max() * 100

print("\nPredicted Job Role:", predicted_role)
print("Job Match Percentage: {:.2f}%".format(match_percentage))
