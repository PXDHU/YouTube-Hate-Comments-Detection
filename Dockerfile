FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Copy the model & vectorizer 
COPY lgbm_model.pkl ./lgbm_model.pkl
COPY tfidf_vectorizer.pkl ./tfidf_vectorizer.pkl

CMD ["python", "flask/main.py"]
