import matplotlib
matplotlib.use('Agg')  

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
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)

# Define the preprocessing function
def preprocess_content(content):
    """Apply preprocessing transformations to a content."""
    try:
        # Convert to lowercase
        content = content.lower()

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        content = ' '.join([lemmatizer.lemmatize(word) for word in content.split()])

        return content
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return content
    
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-44-222-128-31.compute-1.amazonaws.com:5000/")  
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  

# def load_model(model_path, vectorizer_path):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
        
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
      
#         return model, vectorizer
#     except Exception as e:
#         raise

# model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")


@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    print("i am the comment: ",comments)
    print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_content(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # # Convert the sparse matrix to dense format
        X_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=vectorizer.get_feature_names_out()
        )  # Convert to dense array
        
        # Make predictions
        # predictions = model.predict(X_df).tolist()  # Convert to list
        predictions = model.predict(X_df)

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)