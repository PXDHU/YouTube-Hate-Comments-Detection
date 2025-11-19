# YouTube Hate Comments Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-lightblue.svg)
![MLflow](https://img.shields.io/badge/MLflow-3.6.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![AWS](https://img.shields.io/badge/AWS-EC2%2FECR-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)

**An end-to-end machine learning system for detecting hate speech and sentiment analysis in YouTube comments, featuring a Chrome extension, RESTful API, and automated ML pipeline with MLflow integration.**

[Features](#-features) • [Installation](#-installation--setup) • [Usage](#-usage) • [Deployment](#-deployment) • [Architecture](#-architecture--folder-structure)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack & Badges](#-tech-stack--badges)
- [Features](#-features)
- [Architecture & Folder Structure](#-architecture--folder-structure)
- [Installation & Setup](#-installation--setup)
- [Deployment](#-deployment)
- [Usage](#-usage)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License & Credits](#-license--credits)
- [Extra Notes](#-extra-notes)

---

## 🎯 Project Overview

### Name
**YouTube Hate Comments Detection**

### Purpose and Motivation
This project addresses the critical need for automated content moderation on YouTube by providing real-time sentiment analysis and hate speech detection for video comments. The system helps content creators, moderators, and platform administrators identify and manage toxic comments efficiently, fostering healthier online communities.

### Summary of Functionality
The system consists of three main components:
1. **Machine Learning Pipeline**: Automated data ingestion, preprocessing, model training, evaluation, and registration using DVC and MLflow
2. **RESTful API**: Flask-based backend service providing sentiment prediction, comment fetching, and visualization endpoints
3. **Chrome Extension**: Browser extension that integrates seamlessly with YouTube, providing real-time sentiment analysis and visualizations

### Target Users
- **Content Creators**: Monitor comment sentiment on their videos
- **Platform Moderators**: Identify and flag hate speech automatically
- **Researchers**: Study sentiment patterns and hate speech trends
- **Developers**: Integrate sentiment analysis capabilities into their applications

---

## 🛠 Tech Stack & Badges

### Core Technologies

| Category | Technology | Version |
|----------|-----------|---------|
| **Language** | Python | 3.11 |
| **Web Framework** | Flask | 3.1.2 |
| **ML Framework** | LightGBM | 4.6.0 |
| **MLOps** | MLflow | 3.6.0 |
| **Data Versioning** | DVC | 3.53.0 |
| **Containerization** | Docker | Latest |
| **Cloud Platform** | AWS (EC2, ECR) | - |
| **CI/CD** | GitHub Actions | - |
| **Frontend** | Chrome Extension API | Manifest V3 |

### Key Libraries

- **Machine Learning**: `lightgbm`, `scikit-learn`, `numpy`, `pandas`
- **NLP**: `nltk`, `transformers`, `tokenizers`
- **Data Processing**: `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **API**: `flask`, `flask-cors`, `requests`
- **MLOps**: `mlflow`, `dvc`, `dvc-s3`
- **Utilities**: `python-dotenv`, `pyyaml`, `pickle`

### Badges

```
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-lightblue.svg)
![MLflow](https://img.shields.io/badge/MLflow-3.6.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![AWS](https://img.shields.io/badge/AWS-EC2%2FECR-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)
```

---

## ✨ Features

### Core Features

1. **Real-time Sentiment Analysis**
   - Classifies comments into three categories: Positive (1), Neutral (0), Negative (-1)
   - Batch processing support for multiple comments
   - Timestamp-aware predictions for trend analysis

2. **Chrome Extension Integration**
   - Seamless YouTube integration
   - Automatic comment fetching from current video
   - Real-time sentiment visualization
   - Dark-themed, YouTube-inspired UI

3. **Advanced Visualizations**
   - **Pie Charts**: Sentiment distribution visualization
   - **Trend Graphs**: Monthly sentiment percentage over time
   - **Word Clouds**: Most frequent words in comments
   - **Confusion Matrices**: Model performance metrics

4. **Comment Analytics Dashboard**
   - Total comments count
   - Unique commenters identification
   - Average comment length
   - Normalized sentiment score (0-10 scale)

5. **ML Pipeline Automation**
   - Automated data ingestion from external sources
   - Text preprocessing pipeline (lowercasing, stopword removal, lemmatization)
   - TF-IDF feature extraction with configurable n-grams
   - LightGBM model training with hyperparameter tuning
   - Model evaluation with comprehensive metrics
   - MLflow experiment tracking and model registry

6. **RESTful API Endpoints**
   - `/predict` - Basic sentiment prediction
   - `/predict_with_timestamps` - Prediction with temporal data
   - `/fetch_comments` - YouTube API integration for comment fetching
   - `/generate_chart` - Dynamic pie chart generation
   - `/generate_wordcloud` - Word cloud visualization
   - `/generate_trend_graph` - Time-series sentiment trends
   - `/health` - Health check endpoint

7. **MLOps Features**
   - DVC pipeline for reproducible experiments
   - MLflow model versioning and registry
   - Automated model registration and staging
   - Experiment tracking with parameters and metrics
   - Artifact logging (confusion matrices, models, vectorizers)

8. **CI/CD Integration**
   - Automated testing and linting
   - Docker image building and pushing to AWS ECR
   - Automated deployment to EC2 instances
   - Self-hosted runner support for EC2 deployment

9. **Data Preprocessing**
   - Automatic stopword removal (with sentiment-preserving exceptions)
   - Text normalization and cleaning
   - Lemmatization for word standardization
   - Special character handling
   - Duplicate and empty comment removal

10. **Model Features**
    - Multi-class classification (3 sentiment classes)
    - TF-IDF vectorization with configurable features (default: 1000)
    - N-gram support (1-3 grams)
    - Class balancing for imbalanced datasets
    - Regularization (L1/L2) for overfitting prevention

---

## 🏗 Architecture & Folder Structure

### System Architecture

```
┌─────────────────┐
│  Chrome Browser │
│   (Extension)   │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   Flask API     │
│  (Port 5000)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│ LightGBM│ │ YouTube │
│  Model  │ │   API   │
└────────┘ └──────────┘
```

### ML Pipeline Architecture

```
Data Ingestion → Preprocessing → Model Building → Evaluation → Registration
     │               │                 │              │             │
     ▼               ▼                 ▼              ▼             ▼
  DVC Stage      DVC Stage        DVC Stage      DVC Stage    MLflow Registry
```

### Folder Structure

```
YouTube-Hate-Comments-Detection/
│
├── .github/
│   └── workflows/
│       └── cicd.yaml              # CI/CD pipeline configuration
│
├── .dvc/                          # DVC configuration and cache
│
├── data/
│   ├── raw/                       # Raw data files
│   │   ├── train.csv
│   │   └── test.csv
│   └── interim/                   # Processed data files
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── flask/                         # Flask application
│   ├── main.py                    # Main Flask app with all routes
│   └── test.py                    # Model testing script
│
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── Baseline_Model_exp_1.ipynb
│   ├── Baseline_Model_exp_3_tfidf_(1,2)_max_features.ipynb
│   └── HateSpeech_preprocessing_EDA.ipynb
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── datapreprocess/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   └── data_preprocessing.py   # Text preprocessing pipeline
│   └── model/
│       ├── model_building.py      # LightGBM model training
│       ├── model_evaluation.py    # Model evaluation and MLflow logging
│       └── register_model.py      # MLflow model registration
│
├── yt-Chrome-plugin-frontend/     # Chrome extension
│   ├── manifest.json              # Extension manifest (Manifest V3)
│   ├── popup.html                 # Extension popup UI
│   └── popup.js                   # Extension logic and API calls
│
├── .dvcignore                     # DVC ignore patterns
├── .gitignore                     # Git ignore patterns
├── Dockerfile                     # Docker container configuration
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC lock file (pipeline state)
├── params.yaml                    # Model and pipeline hyperparameters
├── requirements.txt               # Python dependencies
├── setup.py                       # Python package setup
├── experiment_info.json           # MLflow experiment metadata
├── lgbm_model.pkl                 # Trained LightGBM model
├── tfidf_vectorizer.pkl           # Trained TF-IDF vectorizer
├── LICENSE                        # MIT License
└── README.md                      # This file
```

### File Descriptions

#### Configuration Files
- **`params.yaml`**: Centralized hyperparameter configuration
  - `data_ingestion.test_size`: Test set split ratio (0.20)
  - `model_building.ngram_range`: TF-IDF n-gram range [1, 3]
  - `model_building.max_features`: Maximum TF-IDF features (1000)
  - `model_building.learning_rate`: LightGBM learning rate (0.09)
  - `model_building.max_depth`: Tree max depth (20)
  - `model_building.n_estimators`: Number of trees (367)

- **`dvc.yaml`**: DVC pipeline stages definition
  - `data_ingestion`: Downloads and splits data
  - `data_preprocessing`: Text normalization and cleaning
  - `model_building`: Feature engineering and model training
  - `model_evaluation`: Model evaluation and MLflow logging
  - `model_registration`: MLflow model registry integration

- **`.github/workflows/cicd.yaml`**: GitHub Actions workflow
  - Integration tests
  - Docker build and push to ECR
  - EC2 deployment automation

#### Source Code Files

**`flask/main.py`**: Main Flask application
- Route handlers for all API endpoints
- Model loading (MLflow or local)
- Text preprocessing functions
- Visualization generation (charts, word clouds, trends)

**`src/datapreprocess/data_ingestion.py`**:
- Loads data from external URL
- Handles missing values and duplicates
- Splits data into train/test sets
- Saves to `data/raw/`

**`src/datapreprocess/data_preprocessing.py`**:
- Text normalization (lowercase, strip)
- Stopword removal (preserves sentiment words)
- Lemmatization
- Special character handling
- Saves to `data/interim/`

**`src/model/model_building.py`**:
- TF-IDF vectorization
- LightGBM classifier training
- Model serialization
- Hyperparameter application from `params.yaml`

**`src/model/model_evaluation.py`**:
- Model performance evaluation
- Classification metrics (precision, recall, F1)
- Confusion matrix generation
- MLflow experiment tracking
- Artifact logging

**`src/model/register_model.py`**:
- MLflow model registration
- Model versioning
- Stage transitions (Staging/Production)

**`yt-Chrome-plugin-frontend/popup.js`**:
- YouTube URL parsing
- Comment fetching via API
- Sentiment prediction requests
- Visualization rendering
- Metrics calculation

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- AWS Account (for cloud deployment)
- YouTube Data API v3 Key (for comment fetching)
- Git

### Local Development Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/YouTube-Hate-Comments-Detection.git
cd YouTube-Hate-Comments-Detection
```

#### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download NLTK Resources

```bash
python -m nltk.downloader stopwords wordnet
```

#### 5. Environment Variables Setup

Create a `.env` file in the root directory:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000  # Optional, for MLflow integration
AWS_ACCESS_KEY_ID=your_aws_access_key  # Optional, for AWS deployment
AWS_SECRET_ACCESS_KEY=your_aws_secret_key  # Optional
AWS_REGION=us-east-1  # Optional
```

**Getting a YouTube API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Copy the key to `.env`

#### 6. Run DVC Pipeline (Optional - for model training)

```bash
# Install DVC (if not already installed)
pip install dvc dvc-s3

# Run the complete pipeline
dvc repro

# Or run individual stages
dvc run data_ingestion
dvc run data_preprocessing
dvc run model_building
dvc run model_evaluation
dvc run model_registration
```

**Note**: The pipeline requires:
- Internet connection for data ingestion
- MLflow server running (for evaluation and registration stages)
- Model files (`lgbm_model.pkl`, `tfidf_vectorizer.pkl`) should be present for API usage

#### 7. Run Flask Application Locally

```bash
# From project root
python flask/main.py
```

The API will be available at `http://localhost:5000`

#### 8. Install Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `yt-Chrome-plugin-frontend/` directory
5. The extension icon should appear in your toolbar

**Note**: Update `API_URL` in `popup.js` if running locally:
```javascript
const API_URL = 'http://localhost:5000';  // Change from production URL
```

### Verify Installation

1. **Test API Health**:
   ```bash
   curl http://localhost:5000/health
   # Expected: {"status": "ok"}
   ```

2. **Test Prediction**:
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"comments": ["I love this video!", "This is terrible"]}'
   ```

3. **Test Chrome Extension**:
   - Navigate to any YouTube video
   - Click the extension icon
   - Verify comment analysis appears

---

## 🚢 Deployment

### Docker Deployment

#### Build Docker Image

```bash
docker build -t youtube-hate-detection:latest .
```

#### Run Docker Container

```bash
docker run -d \
  --name youtube-hate-detection \
  -p 5000:5000 \
  -e YOUTUBE_API_KEY=your_api_key \
  youtube-hate-detection:latest
```

#### Verify Container

```bash
docker ps
docker logs youtube-hate-detection
curl http://localhost:5000/health
```

### AWS Deployment (EC2 + ECR)

#### Prerequisites
- AWS CLI configured
- EC2 instance with Docker installed
- ECR repository created
- GitHub Secrets configured (see CI/CD section)

#### Manual Deployment Steps

1. **Build and Push to ECR**:
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build image
   docker build -t youtube-hate-detection .
   
   # Tag image
   docker tag youtube-hate-detection:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/youtube-hate-detection:latest
   
   # Push to ECR
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/youtube-hate-detection:latest
   ```

2. **Deploy on EC2**:
   ```bash
   # SSH into EC2 instance
   ssh -i your-key.pem ec2-user@your-ec2-ip
   
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Pull latest image
   docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/youtube-hate-detection:latest
   
   # Stop existing container
   docker stop youtube-hate-detection || true
   docker rm youtube-hate-detection || true
   
   # Run new container
   docker run -d \
     --name youtube-hate-detection \
     -p 5000:5000 \
     -e YOUTUBE_API_KEY=your_api_key \
     <account-id>.dkr.ecr.us-east-1.amazonaws.com/youtube-hate-detection:latest
   ```

3. **Configure Security Groups**:
   - Open port 5000 in EC2 security group
   - Allow inbound traffic from your IP or 0.0.0.0/0 (for public access)

### CI/CD Workflow

The project includes automated CI/CD via GitHub Actions. The workflow (`.github/workflows/cicd.yaml`) performs:

1. **Integration Tests**: Runs linting and unit tests
2. **Docker Build**: Builds Docker image
3. **ECR Push**: Pushes image to AWS ECR
4. **EC2 Deployment**: Deploys to EC2 using self-hosted runner

#### GitHub Secrets Configuration

Configure these secrets in your GitHub repository settings:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key for ECR/EC2 access |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (e.g., `us-east-1`) |
| `ECR_REPOSITORY_NAME` | ECR repository name |
| `AWS_ECR_LOGIN_URI` | ECR login URI (e.g., `123456789.dkr.ecr.us-east-1.amazonaws.com`) |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key |

#### Self-Hosted Runner Setup (EC2)

1. **Install GitHub Actions Runner on EC2**:
   ```bash
   # Download runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
   
   # Configure runner
   ./config.sh --url https://github.com/yourusername/yourrepo --token YOUR_TOKEN
   
   # Install as service
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

2. **Label the Runner**:
   - In GitHub repo: Settings → Actions → Runners
   - Label the runner (e.g., `self-hosted`)

3. **Update Workflow** (if needed):
   ```yaml
   runs-on: self-hosted  # Matches your runner label
   ```

#### Triggering Deployment

Deployment is automatically triggered on:
- Push to `main` branch (excluding README.md changes)
- Manual workflow dispatch (GitHub Actions UI)

### MLflow Server Setup (Optional)

For model tracking and registry:

1. **Install MLflow**:
   ```bash
   pip install mlflow
   ```

2. **Start MLflow Server**:
   ```bash
   mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns \
     --host 0.0.0.0 \
     --port 5000
   ```

3. **Update MLflow URI**:
   - In `src/model/model_evaluation.py`: Update `mlflow.set_tracking_uri()`
   - In `src/model/register_model.py`: Update `mlflow.set_tracking_uri()`
   - In Flask app: Set `mlflow_uri` parameter in `get_model()`

### Redeploying Updates

#### Automated (Recommended)
- Push changes to `main` branch
- GitHub Actions automatically builds and deploys

#### Manual
1. Make code changes
2. Commit and push
3. Or manually trigger workflow in GitHub Actions UI

---

## 📖 Usage

### API Usage

#### 1. Health Check

```bash
curl http://localhost:5000/health
```

**Response**:
```json
{"status": "ok"}
```

#### 2. Basic Sentiment Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "I love this video!",
      "This is terrible",
      "Neutral comment here"
    ]
  }'
```

**Response**:
```json
[
  {"comment": "I love this video!", "sentiment": "1"},
  {"comment": "This is terrible", "sentiment": "-1"},
  {"comment": "Neutral comment here", "sentiment": "0"}
]
```

**Sentiment Labels**:
- `1`: Positive
- `0`: Neutral
- `-1`: Negative

#### 3. Prediction with Timestamps

```bash
curl -X POST http://localhost:5000/predict_with_timestamps \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      {"text": "Great video!", "timestamp": "2024-01-15T10:30:00Z"},
      {"text": "Not good", "timestamp": "2024-01-15T11:00:00Z"}
    ]
  }'
```

**Response**:
```json
[
  {"comment": "Great video!", "sentiment": "1", "timestamp": "2024-01-15T10:30:00Z"},
  {"comment": "Not good", "sentiment": "-1", "timestamp": "2024-01-15T11:00:00Z"}
]
```

#### 4. Fetch YouTube Comments

```bash
curl -X POST http://localhost:5000/fetch_comments \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ"
  }'
```

**Response**: YouTube API response with comment threads

#### 5. Generate Pie Chart

```bash
curl -X POST http://localhost:5000/generate_chart \
  -H "Content-Type: application/json" \
  -d '{
    "sentiment_counts": {
      "1": 50,
      "0": 30,
      "-1": 20
    }
  }' \
  --output chart.png
```

**Response**: PNG image file

#### 6. Generate Word Cloud

```bash
curl -X POST http://localhost:5000/generate_wordcloud \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "This is a great video",
      "I love the content",
      "Amazing work"
    ]
  }' \
  --output wordcloud.png
```

**Response**: PNG image file

#### 7. Generate Trend Graph

```bash
curl -X POST http://localhost:5000/generate_trend_graph \
  -H "Content-Type: application/json" \
  -d '{
    "sentiment_data": [
      {"timestamp": "2024-01-01T00:00:00Z", "sentiment": 1},
      {"timestamp": "2024-02-01T00:00:00Z", "sentiment": 0},
      {"timestamp": "2024-03-01T00:00:00Z", "sentiment": -1}
    ]
  }' \
  --output trend.png
```

**Response**: PNG image file

### Chrome Extension Usage

1. **Navigate to YouTube**: Open any YouTube video
2. **Open Extension**: Click the extension icon in Chrome toolbar
3. **View Analysis**: The extension automatically:
   - Fetches comments from the current video
   - Performs sentiment analysis
   - Displays metrics and visualizations
4. **Review Results**:
   - Comment Analysis Summary (total comments, unique commenters, averages)
   - Sentiment Distribution (pie chart)
   - Sentiment Trend Over Time (line graph)
   - Word Cloud
   - Top 25 Comments with Sentiments

### Python SDK Usage

```python
import requests

API_URL = "http://localhost:5000"

# Predict sentiment
response = requests.post(
    f"{API_URL}/predict",
    json={"comments": ["I love this!", "This is bad"]}
)
predictions = response.json()
print(predictions)
```

### DVC Pipeline Usage

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_building

# View pipeline graph
dvc dag

# Check pipeline status
dvc status

# Update parameters and rerun
# Edit params.yaml, then:
dvc repro
```

---

## 🧪 Testing

### Unit Testing

Currently, the project includes placeholder test commands in the CI/CD workflow. To add comprehensive tests:

1. **Create test directory**:
   ```bash
   mkdir tests
   ```

2. **Example test file** (`tests/test_api.py`):
   ```python
   import pytest
   from flask import Flask
   from flask.main import app
   
   @pytest.fixture
   def client():
       app.config['TESTING'] = True
       with app.test_client() as client:
           yield client
   
   def test_health_endpoint(client):
       response = client.get('/health')
       assert response.status_code == 200
       assert response.json == {"status": "ok"}
   
   def test_predict_endpoint(client):
       response = client.post('/predict', json={
           "comments": ["test comment"]
       })
       assert response.status_code == 200
   ```

3. **Run tests**:
   ```bash
   pip install pytest pytest-cov
   pytest tests/ -v
   pytest tests/ --cov=flask --cov=src
   ```

### Integration Testing

Test the complete pipeline:

```bash
# Test data ingestion
python src/datapreprocess/data_ingestion.py

# Test preprocessing
python src/datapreprocess/data_preprocessing.py

# Test model building
python src/model/model_building.py

# Test evaluation
python src/model/model_evaluation.py
```

### Manual API Testing

Use the provided test script:

```bash
python flask/test.py
```

### Chrome Extension Testing

1. Load extension in Chrome (Developer mode)
2. Navigate to a YouTube video
3. Open extension popup
4. Verify all features work correctly

### Test Coverage

To generate coverage reports:

```bash
pytest --cov=flask --cov=src --cov-report=html
open htmlcov/index.html  # View report
```

---

## 🤝 Contributing

### Contribution Guidelines

We welcome contributions! Please follow these guidelines:

1. **Fork the Repository**: Create your own fork
2. **Create a Feature Branch**: 
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Follow Code Style**:
   - Use PEP 8 for Python code
   - Add docstrings to functions
   - Include type hints where possible
4. **Write Tests**: Add tests for new features
5. **Update Documentation**: Update README if needed
6. **Commit Changes**:
   ```bash
   git commit -m "Add: description of changes"
   ```
7. **Push and Create Pull Request**

### Branching Strategy

- **`main`**: Production-ready code
- **`develop`**: Development branch (if used)
- **`feature/*`**: Feature branches
- **`bugfix/*`**: Bug fix branches
- **`hotfix/*`**: Critical production fixes

### Code Review Process

1. All PRs require at least one review
2. CI/CD checks must pass
3. Code should be well-documented
4. Tests should be included

### Development Setup

1. Follow [Installation & Setup](#-installation--setup)
2. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```
3. Run linters:
   ```bash
   black flask/ src/ --check
   flake8 flask/ src/
   mypy flask/ src/
   ```

---

## 📄 License & Credits

### License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

### Author

**Padmavasan Balakrishnan**

- Email: padmavasan.contact@gmail.com
- GitHub: [@PXDHU](https://github.com/PXDHU)
- LinkedIn: [Padmavasan Balakrishnan](https://www.linkedin.com/in/padmavasanbalakrishnan/)

### Acknowledgments

- **Data Source**: Reddit sentiment analysis dataset (used for training)
- **YouTube Data API**: For comment fetching functionality
- **Open Source Libraries**: All contributors to the libraries used in this project
- **MLflow Community**: For excellent MLOps tooling
- **DVC Team**: For data version control capabilities

### Third-Party Services

- **YouTube Data API v3**: Comment fetching
- **AWS (EC2, ECR)**: Cloud infrastructure
- **GitHub Actions**: CI/CD automation

---

## 📝 Extra Notes

### Security Considerations

1. **API Key Protection**:
   - Never commit `.env` files
   - Use environment variables in production
   - Rotate API keys regularly
   - Use AWS Secrets Manager for production

2. **Docker Security**:
   - Run containers as non-root user (add to Dockerfile)
   - Keep base images updated
   - Scan images for vulnerabilities

3. **API Security**:
   - Implement rate limiting (consider Flask-Limiter)
   - Add authentication for production (API keys, JWT)
   - Use HTTPS in production
   - Validate and sanitize all inputs

4. **CORS Configuration**:
   - Currently allows all origins (`CORS(app)`)
   - Restrict to specific domains in production:
     ```python
     CORS(app, resources={r"/*": {"origins": ["https://yourdomain.com"]}})
     ```

### Known Issues & Limitations

1. **Model Limitations**:
   - Trained on Reddit data, may not generalize perfectly to YouTube comments
   - Limited to English language
   - May misclassify sarcasm or context-dependent comments

2. **API Limitations**:
   - YouTube API has quota limits (10,000 units/day default)
   - Batch processing limited by memory
   - No authentication/authorization implemented

3. **Chrome Extension**:
   - Requires manual installation (not in Chrome Web Store)
   - API URL hardcoded (update for different environments)
   - Limited to YouTube.com domain

4. **MLflow Integration**:
   - Requires MLflow server to be running
   - Model registration may fail if server is unavailable
   - No automatic fallback mechanism

5. **Data Pipeline**:
   - Data ingestion depends on external URL availability
   - No data validation schema
   - Limited error recovery

### Future Improvements

1. **Model Enhancements**:
   - Fine-tune on YouTube-specific data
   - Support for multiple languages
   - Context-aware sentiment analysis
   - Real-time model retraining pipeline

2. **API Improvements**:
   - Authentication and authorization
   - Rate limiting
   - Request caching
   - GraphQL API option
   - WebSocket support for real-time updates

3. **Frontend Enhancements**:
   - Publish Chrome extension to Web Store
   - Add Firefox extension support
   - Real-time comment streaming
   - Customizable visualization themes

4. **MLOps Improvements**:
   - Automated model retraining on schedule
   - A/B testing framework
   - Model performance monitoring
   - Automated rollback on performance degradation

5. **Infrastructure**:
   - Kubernetes deployment
   - Auto-scaling based on load
   - Multi-region deployment
   - CDN for static assets

6. **Features**:
   - Comment filtering and moderation actions
   - User reputation scoring
   - Sentiment trend alerts
   - Export functionality (CSV, PDF reports)

### Performance Optimization

1. **Model Optimization**:
   - Model quantization for faster inference
   - Batch prediction optimization
   - Caching frequent predictions

2. **API Optimization**:
   - Redis caching for predictions
   - Database for comment storage
   - Async request handling (consider FastAPI migration)

3. **Extension Optimization**:
   - Lazy loading of visualizations
   - Progressive data fetching
   - Client-side caching

### Monitoring & Logging

1. **Application Logging**:
   - Currently uses Python logging
   - Consider structured logging (JSON format)
   - Integration with logging services (CloudWatch, Datadog)

2. **Metrics Collection**:
   - Request/response times
   - Error rates
   - Model prediction latency
   - API usage statistics

3. **Alerting**:
   - Set up alerts for API failures
   - Model performance degradation alerts
   - Resource usage alerts

### Documentation

- Consider adding API documentation (Swagger/OpenAPI)
- Add architecture diagrams
- Create video tutorials
- Add Jupyter notebook tutorials

---

## 📞 Support

For issues, questions, or contributions:

1. **Open an Issue**: Use GitHub Issues for bug reports and feature requests
2. **Contact Author**: Email padmavasan.contact@gmail.com
3. **Check Documentation**: Review this README and code comments

---

<div align="center">

**Made with ❤️ by Padmavasan Balakrishnan**

[GitHub](https://github.com/PXDHU) • [LinkedIn](https://www.linkedin.com/in/padmavasanbalakrishnan/)

⭐ Star this repo if you find it useful!

</div>
