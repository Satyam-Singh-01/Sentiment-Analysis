import streamlit as st
import textblob
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text data
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

# Function to train SVM and Random Forest models and display their accuracies
def train_models(data_file):
    data = pd.read_csv(data_file)
    data['Sentences'] = data['Sentences'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(data['Sentences'], data['Sentiment'], test_size=0.2, random_state=400)

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train SVM model
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_classifier.predict(X_test_tfidf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    svm_report = classification_report(y_test, y_pred_svm)

    # Train Random Forest model
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_tfidf, y_train)
    y_pred_rf = rf_classifier.predict(X_test_tfidf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf)

    return accuracy_svm, svm_report, accuracy_rf, rf_report

# Function to analyze sentiment using TextBlob
def analyze_single_comment(comment):
    analysis = textblob.TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        st.success("Review: Positive")
    elif polarity == 0:
        st.success("Review: Neutral")
    else:
        st.success("Review: Negative")

# Function to check if the comment contains any letters
def is_valid_comment(comment):
    return bool(re.search(r'[a-zA-Z]', comment))

# Streamlit app
st.title("Sentiment Analysis")

# Display accuracies of SVM and Random Forest models
if st.checkbox("Show Model Accuracies"):
    accuracy_svm, svm_report, accuracy_rf, rf_report = train_models('classified.csv')
    st.write(f"SVM Accuracy: {accuracy_svm}")
    st.write("SVM Classification Report:")
    st.text(svm_report)
    st.write(f"Random Forest Accuracy: {accuracy_rf}")
    st.write("Random Forest Classification Report:")
    st.text(rf_report)

# Text input and analysis
comment = st.text_input("Enter any comment")
b = st.button("Analyze Comment")

if "select" not in st.session_state:
    st.session_state["select"] = False

if not st.session_state["select"]:
    if b:
        if is_valid_comment(comment):
            analyze_single_comment(comment)
        else:
            st.error("Incorrect format, Please enter a valid comment (must contain letters)")
else:
    st.error("Incorrect format, Please enter a valid comment")
