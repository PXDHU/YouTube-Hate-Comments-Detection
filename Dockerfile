# Use Debian-based slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential libffi-dev wget curl git && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python3 -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python3", "flask/main.py"]
