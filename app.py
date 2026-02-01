import streamlit as st
import pickle

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("spam_bow_nb.pkl", "rb") as f:
        model = pickle.load(f)

    with open("bow_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam Detection", layout="centered")

st.title("📩 SMS Spam Detection")
st.write("Enter a message and the model will predict whether it is **Spam** or **Ham**.")

user_input = st.text_area(
    "Message",
    placeholder="Type or paste your message here...",
    height=150
)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if prediction == 1:
            st.error(f"🚨 Spam\n\nConfidence: {prob[1]:.2%}")
        else:
            st.success(f"✅ Ham (Not Spam)\n\nConfidence: {prob[0]:.2%}")
