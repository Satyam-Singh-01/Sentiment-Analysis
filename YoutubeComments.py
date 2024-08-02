import streamlit as st
from googleapiclient.discovery import build
from langdetect import detect
import textblob
import re

api_key = 'Your_API_Key'


def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        analysis = textblob.TextBlob(comment)
        sentiments.append(analysis.sentiment.polarity)
    return sentiments

def extract_youtube_video_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()
    comments = []
    while len(comments) <= 100:
        for item in video_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            try:
                if detect(comment) == 'en':
                    comments.append(comment)
            except:
                pass

        # Handle pagination
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken']
            ).execute()
        else:
            break
    return comments

def analyze_yt_comments(video_id):
    comments = extract_youtube_video_comments(video_id)

    sentiment = analyze_sentiment(comments)
    print("Sentiment Analysis of the video comments:")
    avg_sentiment = sum(sentiment) / len(sentiment)

    # Print overall review
    if avg_sentiment > 0:
        st.success("Overall Review: Positive")
    elif avg_sentiment == 0:
        st.success("Overall Review: Neutral")
    else:
        st.success("Overall Review: Negative")

def extract_video_id(youtube_link):
    # Regular expression pattern to match YouTube video IDs
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_link)
    if match:
        return match.group(1)  # Extract the video ID from the matched pattern
    else:
        return None

def is_valid_youtube_link(link):
    # Check if the link is valid and contains letters
    return bool(re.search(r'[a-zA-Z]', link)) and extract_video_id(link) is not None

st.title("Analyzing Sentiments in Students Reviews of Online Courses")
url = st.text_input("Enter YouTube course video URL")
b = st.button("Analyze Sentiment")

if "select" not in st.session_state:
    st.session_state["select"] = False

if not st.session_state["select"]:
    if b:
        if is_valid_youtube_link(url):
            id = extract_video_id(url)
            analyze_yt_comments(id)
        else:
            st.error("Invalid URL, Please enter a valid video URL (must contain letters and be a valid YouTube link)")
else:
    st.error("Invalid URL, Please enter a valid video URL")
