import streamlit as st
import pickle
import pandas as pd
import re
import string

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔍 Prediction"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("📰 Fake News Detection System")

    st.success("🎯 Model Accuracy: 98.19%")

    st.info(
        """
        This project detects fake news articles using
        Natural Language Processing (NLP),
        TF-IDF Vectorization and Logistic Regression.
        """
    )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "98.19%")
    col2.metric("Precision", "98%")
    col3.metric("Recall", "98%")
    col4.metric("F1 Score", "98%")

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")

    st.write("📄 Total Records: 44,898")
    st.write("❌ Fake News Articles: 23,481")
    st.write("✅ Real News Articles: 21,417")

    st.markdown("---")

    st.subheader("📈 Model Evaluation")

    st.code("""
Confusion Matrix

[[395 10]
 [  5 390]]
""")

    st.code("""
Classification Report

Precision : 98%
Recall    : 98%
F1 Score  : 98%

Accuracy  : 98.19%
""")

    st.markdown("---")

    st.subheader("⚙️ Project Workflow")

    st.markdown("""
1. Collect Dataset

2. Perform Data Cleaning

3. Apply NLP Preprocessing

4. Convert Text using TF-IDF

5. Train Logistic Regression Model

6. Evaluate Model Performance

7. Save Model (.pkl)

8. Deploy using Streamlit
""")

    st.markdown("---")

    st.subheader("🚀 Project Features")

    st.success("""
✔ Fake News Detection

✔ Real News Detection

✔ NLP Text Processing

✔ TF-IDF Vectorization

✔ Logistic Regression Classification

✔ Confidence Score Prediction

✔ Interactive Streamlit Dashboard
""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    st.info("""
Python | Pandas | NumPy | NLP | Scikit-Learn

TF-IDF | Logistic Regression | Streamlit

Matplotlib | Seaborn | WordCloud
""")

    st.markdown("---")

    st.markdown("""
### 👨‍💻 Developed By

Lakshmi Priya Vellanki

M.Tech – AI & Data Science

Fake News Detection using NLP and Machine Learning
""")


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "🔍 Prediction":

    st.title("🔍 News Prediction")

    st.markdown(
        "Enter a news article below and click Predict."
    )

    st.subheader("🧪 Sample News")

    st.code(
        """
The Reserve Bank of India announced new measures to strengthen the banking sector. The policy includes revised lending guidelines and increased support for digital payment infrastructure.
"""
    )

    news_text = st.text_area(
        "News Article Text",
        height=250
    )

    if st.button("🚀 Predict News"):

        if news_text.strip() == "":

            st.warning(
                "Please enter some news text."
            )

        else:

            cleaned_text = clean_text(news_text)

            transformed_text = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            probability = model.predict_proba(
                transformed_text
            )

            confidence = max(
                probability[0]
            ) * 100

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            if prediction == 0:

                st.error(
                    "🚨 FAKE NEWS DETECTED"
                )

            else:

                st.success(
                    "✅ REAL NEWS DETECTED"
                )

            st.info(
                f"Confidence Score: {confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            st.markdown("---")

            st.subheader(
                "📈 Confidence Analysis"
            )

            fake_prob = probability[0][0] * 100
            real_prob = probability[0][1] * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fake News %",
                    f"{fake_prob:.2f}%"
                )

            with col2:
                st.metric(
                    "Real News %",
                    f"{real_prob:.2f}%"
                )

            st.markdown("---")

            st.subheader(
                "📊 Prediction Probability Chart"
            )

            chart_data = pd.DataFrame(
                {
                    "Probability": [
                        fake_prob,
                        real_prob
                    ]
                },
                index=[
                    "Fake News",
                    "Real News"
                ]
            )

            st.bar_chart(chart_data)

            st.markdown("---")

            st.subheader(
                "📝 Processed Text Preview"
            )

            st.write(
                cleaned_text[:500]
            )

            st.markdown("---")

            st.success(
                "Prediction completed successfully using TF-IDF Vectorization and Logistic Regression."
            )

