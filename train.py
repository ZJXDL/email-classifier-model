import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Ensure the 'model' directory exists
os.makedirs('model', exist_ok=True)

# Load the data with encoding fix
df = pd.read_csv("emails.csv", encoding='latin1')

# Rename columns for clarity
df = df.rename(columns={"v1": "label", "v2": "text"})

# Build the model pipeline: TF-IDF + Naive Bayes
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Train the model
model.fit(df['text'], df['label'])

# Save the trained model
joblib.dump(model, 'model/email_model.pkl')

print("✅ Model trained and saved to 'model/email_model.pkl'!")
