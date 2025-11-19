import matplotlib
matplotlib.use('Agg')  # non-interactive backend

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging
import pickle
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import os
from dotenv import load_dotenv
load_dotenv()
import requests

# downloads (only once at runtime; ensure server has internet on first run)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)  # allow all origins by default (adjust if needed)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals for lazy loading
mlflow_model = None
mlflow_vectorizer = None

# ---------- Preprocessing ----------
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = (comment or "").lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^\w\s!?.,]', '', comment, flags=re.UNICODE)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        logger.exception("Error in preprocess_comment")
        return comment

# ---------- Model load helpers ----------
def load_model_and_vectorizer_from_mlflow(model_name, model_version, vectorizer_path, mlflow_uri=None):
    """Load an MLflow pyfunc model and a local vectorizer pickle."""
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def load_model_and_vectorizer_from_disk(model_path, vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def get_model(mlflow_uri=None, mlflow_name="yt_chrome_plugin_model", mlflow_version="2", vectorizer_path="./tfidf_vectorizer.pkl", local_model_path=None):
    """Lazy-load model/vectorizer into global variables. Returns (model, vectorizer)."""
    global mlflow_model, mlflow_vectorizer
    if mlflow_model is not None and mlflow_vectorizer is not None:
        return mlflow_model, mlflow_vectorizer

    try:
        logger.info("Attempting to load model from MLflow (if configured)...")
        if mlflow_uri:
            mlflow_model, mlflow_vectorizer = load_model_and_vectorizer_from_mlflow(mlflow_name, mlflow_version, vectorizer_path, mlflow_uri)
            logger.info("Loaded model from MLflow.")
        elif local_model_path:
            mlflow_model, mlflow_vectorizer = load_model_and_vectorizer_from_disk(local_model_path, vectorizer_path)
            logger.info("Loaded model from disk.")
        else:
            # try disk fallback
            mlflow_model, mlflow_vectorizer = load_model_and_vectorizer_from_disk("./lgbm_model.pkl", vectorizer_path)
            logger.info("Loaded model from default disk location.")
        return mlflow_model, mlflow_vectorizer
    except Exception as e:
        logger.exception("Failed to load model/vectorizer")
        raise

# ---------- Routes ----------
@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.get_json(force=True)
    comments_data = data.get('comments')
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # lazy load model (adjust mlflow_uri/local_model_path as needed)
        model, vectorizer = get_model(
            mlflow_uri=None,               # <--- set to MLflow URI string if you want MLflow
            mlflow_name="yt_chrome_plugin_model",
            mlflow_version="2",
            vectorizer_path="./tfidf_vectorizer.pkl",
            local_model_path="./lgbm_model.pkl"  # optional disk fallback
        )

        comments = [item['text'] for item in comments_data]
        timestamps = [item.get('timestamp') for item in comments_data]

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        # many sklearn models accept sparse input; try using sparse directly
        try:
            preds = model.predict(transformed_comments)
        except Exception:
            # some models expect dense array
            preds = model.predict(transformed_comments.toarray())
        preds = [str(p) for p in preds]

        response = [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, preds, timestamps)]
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        model, vectorizer = get_model(local_model_path="./lgbm_model.pkl", vectorizer_path="./tfidf_vectorizer.pkl")
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed = vectorizer.transform(preprocessed_comments)
        # If model expects DataFrame, convert; otherwise pass sparse/dense accordingly
        try:
            preds = model.predict(transformed)
        except Exception:
            preds = model.predict(transformed.toarray())
        preds = [str(p) for p in preds]
        response = [{"comment": c, "sentiment": s} for c, s in zip(comments, preds)]
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500



@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
    
@app.route("/fetch_comments", methods=["POST"])
def fetch_comments_route():
    data = request.json
    video_id = data.get("video_id")
    
    api_key = os.getenv("YOUTUBE_API_KEY")  # load from .env
    
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "key": api_key
    }

    r = requests.get(url, params=params)
    return jsonify(r.json())


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)