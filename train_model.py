import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("career_recommendation_dataset.csv")

# Ensure stopwords are available
nltk.download("stopwords")

# Function to preprocess job descriptions
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  
    return text

# Apply preprocessing to job descriptions
df["processed_skills"] = df["skills"].apply(preprocess)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
skills_matrix = vectorizer.fit_transform(df["processed_skills"])

# Save the trained vectorizer and skills matrix
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(skills_matrix, "skills_matrix.pkl")

print("✅ Training completed. Files saved successfully!")
