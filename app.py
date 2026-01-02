import streamlit as st
import tensorflow as tf
import numpy as np

# 1. የገጹ አቀማመጥ (Professional UI Design)
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="centered")

# በጎን በኩል መረጃ ለማሳየት (Sidebar)
with st.sidebar:
    st.title("Settings & Info")
    st.info("This AI uses a BiLSTM neural network to analyze the sentiment of your text.")
    st.markdown("---")
    st.write("📊 **Model Status:** Ready")
    st.caption("Developed for Amharic & English text.")

# ዋናው ርዕስ
st.title("🧠 Sentiment Analysis System")
st.markdown("Enter your text below to analyze its sentiment (Positive or Negative).")

# 2. ሞዴሉን መጫን
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("sentiment_bilstm_model.keras")
        vec_model = tf.keras.models.load_model("vectorizer.keras")
        return model, vec_model.layers[0]
    except:
        return None, None

model, vectorizer = load_assets()

if model is None:
    st.error("❌ Error: Could not load model files. Please check your GitHub repository.")
else:
    # 3. የጽሑፍ ግብዓት (Placeholder ያለ አማርኛ ምሳሌ)
    user_text = st.text_area("Your Text:", 
                             placeholder="Type your comment here...",
                             height=150)

    if st.button("Analyze Sentiment"):
        if user_text.strip():
            with st.spinner('Processing...'):
                vec_text = vectorizer([user_text])
                prediction = model.predict(vec_text, verbose=0)[0][0]

            st.divider()

            # ውጤት ማሳያ
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction >= 0.5:
                    st.success("### 😊 Positive")
                    st.balloons()
                else:
                    st.error("### 😞 Negative")

            with col2:
                st.metric(label="Confidence Score", value=f"{prediction:.2%}")
                st.progress(float(prediction))

        else:
            st.warning("⚠️ Please enter some text first.")

st.markdown("---")
st.caption("© 2024 AI Sentiment Analyzer")
