import streamlit as st
import tensorflow as tf
import numpy as np

# 1. የገጹ አቀማመጥ እና ዲዛይን
st.set_page_config(page_title="የስሜት ትንተና",page_icon="", layout="centered")

# --- የጎን ሜኑ (Sidebar) ---
with st.sidebar:
    st.title(" About Project(ስለ ፕሮጀክቱ)")
    st.info("ይህ መተግበሪያ የሰው ሰራሽ አስተውሎት (Deep Learning) ቴክኖሎጂን በመጠቀም የተጻፉ ጽሑፎችን ስሜት ይተነትናል።")
    st.markdown("---")
    st.write(" **ፋይሎች:**")
    st.write("- sentiment_bilstm_model.keras")
    st.write("- vectorizer.keras")
    st.caption("በ BiLSTM ሞዴል የተገነባ።")

# --- ዋናው ገጽ ---
st.title("🧠 Sentiment Analysis System")
st.write("የሚሰማዎትን ወይም ያነበቡትን ጽሑፍ ከታች ባለው ሳጥን ውስጥ ያስገቡ።")

# ሞዴሉን መጫን
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("sentiment_bilstm_model.keras")
        vec_model = tf.keras.models.load_model("vectorizer.keras")
        return model, vec_model.layers[0]
    except Exception as e:
        return None, str(e)

model, vectorizer = load_assets()

if model is None:
    st.error(f"❌ ሞዴሉን መጫን አልተቻለም፦ {vectorizer}")
else:
    # የጽሑፍ ግብዓት
    user_text = st.text_area("ጽሑፍ እዚህ ይጻፉ:", placeholder="ለምሳሌ፦ ምርቱ በጣም ጥሩ ነው...", height=150)

    if st.button("ተንትን (Analyze)"):
        if user_text.strip():
            # ትንተና
            vec_text = vectorizer([user_text])
            prediction = model.predict(vec_text, verbose=0)[0][0]

            st.divider()

            # --- ውጤት በውበት (Styling) ማሳያ ---
            if prediction >= 0.5:
                # ለአዎንታዊ ውጤት አረንጓዴ (Success)
                st.success(f"### 🤷‍♀️ ውጤት፦ አዎንታዊ (Positive)")
                st.balloons()
            else:
                # ለአሉታዊ ውጤት ቀይ (Error)
                st.error(f"### 🤦‍♂️ ውጤት፦ አሉታዊ (Negative)")

            # የእርግጠኝነት መጠን (Confidence)
            st.write(f"**የእርግጠኝነት መጠን (Score):** {prediction:.2%}")
            st.progress(float(prediction))
        else:
            st.warning("⚠️ እባክዎ መጀመሪያ ጽሑፍ ያስገቡ።")
st.markdown("---")
# አምዶችን በመጠቀም ሊንኮቹን ጎን ለጎን ማድረግ
#col1, col2, col3 = st.columns([1,1,1])

st.markdown("**Contact Me:**")
    # የቴሌግራም ሊንክ 
st.markdown("Telegram(https://t.me/Animut_embiale,  Facebook(https://web.facebook.com/Animutanch")
st.divider() # ቀጭን መስመር ያስምራል
st.caption("Set by [Animut Embiale,|College of Engineering and Technology |, |Dept of IT |, Injibara University")
st.caption("© January 2026 | All Rights Reserved")
