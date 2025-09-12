import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

df = pd.read_csv("emails.csv")
df = df.rename(columns={"v1": "label", "v2": "text"})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

clf = MultinomialNB()
clf.fit(X, df['label'])

joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
joblib.dump(clf, 'model/classifier.pkl')

print("✅ Model trained and saved!")