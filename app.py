import streamlit as st
import joblib

vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
clf = joblib.load('model/classifier.pkl')

st.title("📧 Smart Email Classifier")
st.write("Paste any email below — I’ll tell you if it’s **spam** or **legit**!")

user_input = st.text_area("Paste email text here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please paste some text!")
    else:
        X = vectorizer.transform([user_input])
        pred = clf.predict(X)[0]
        prob = max(clf.predict_proba(X)[0]) * 100

        if pred == "spam":
            st.error(f"🚨 SPAM! (Confidence: {prob:.1f}%)")
        else:
            st.success(f"✅ LEGIT EMAIL! (Confidence: {prob:.1f}%)")

st.caption("Built with ❤️ by [Your Name] — ML Engineer")