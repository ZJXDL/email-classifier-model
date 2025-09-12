import streamlit as st
import joblib

# Load the trained model
model = joblib.load('model/email_model.pkl')

# Page title
st.title("📧 Smart Email Classifier")
st.write("Paste any email below — I’ll tell you if it’s **spam** or **legit**!")

# Text input box
user_input = st.text_area("Paste email text here:", height=150)

# Button to classify
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please paste some text!")
    else:
        prediction = model.predict([user_input])[0]
        confidence = max(model.predict_proba([user_input])[0]) * 100

        if prediction == "spam":
            st.error(f"🚨 SPAM! (Confidence: {confidence:.1f}%)")
        else:
            st.success(f"✅ LEGIT EMAIL! (Confidence: {confidence:.1f}%)")

# Optional footer
st.caption("Built with ❤️ by [Your Name] — ML Engineer")