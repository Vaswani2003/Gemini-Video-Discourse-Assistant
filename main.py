import streamlit as st
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import sent_tokenize

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi

import google.generativeai as genai


def download_nltk_components():
    nltk.download('punkt')
    nltk.download('stopwords')

def load_test_data_from_text(text):
    documents = text.split("\n\n")
    test_data = [{"document": document} for document in documents]
    
    return test_data

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)


def calculate_tfidf(sentences):
    non_stop_words_exist = any(word for word in preprocess_text(' '.join(sentences)).split() if word)
    if not non_stop_words_exist:
        return {}  # Return an empty dictionary if no non-stop words are found
    
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = dict(zip(sentences, tfidf_matrix.toarray()))
    return tfidf_scores


def select_top_sentences(tfidf_scores, K=5):
    sorted_sentences = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_sentences = [sentence for sentence, score in sorted_sentences[:K]]
    
    return top_k_sentences

def generate_summaries(data):
    summaries = []
    for item in data:
        document_text = item["document"]
        preprocessed_text = clean_text(document_text)
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(preprocessed_text)

        if sentences:
            # Calculate TF-IDF scores
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            tfidf_scores = dict(zip(sentences, np.array(tfidf_matrix.sum(axis=1)).ravel()))
            top_k_sentences = select_top_sentences(tfidf_scores)
            
            summary = ' '.join(top_k_sentences)
            summaries.append(summary)
    return summaries


def main():
    st.write("# Enter Youtube URL")

    video_url = st.text_input("Enter YouTube Video ID:")
    video_id = video_url.split("=")[-1]

    st.write(f"Study Material based on the video {video_id}")

    Prompt = st.text_input("Any special instructions? Leave blank if none")
    if(Prompt is None):
        Prompt = "Go through this transcript and generate elaborate study material based on the video."

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    for transcript in transcript_list:
        text = transcript['text']
        break

    test_data = load_test_data_from_text(text)
    generated_summaries = generate_summaries(test_data)

    genai.configure(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    model = genai.GenerativeModel('gemini-pro')

    generated_results = []
    for summary in generated_summaries:
        result = model.generate_content(Prompt+summary)
        generated_results.append(result)

    for idx, result in enumerate(generated_results, start=1):
        if result.candidates:
            generated_text = result.candidates[0].content.parts[0].text
            st.write(f"{generated_text}")
        


if __name__ == "__main__":
    main()