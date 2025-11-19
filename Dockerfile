# Use official Python Alpine image
FROM python:3.12.12-alpine3.21

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache gcc g++ musl-dev libffi-dev bash

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python3 -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python3", "flask/main.py"]
