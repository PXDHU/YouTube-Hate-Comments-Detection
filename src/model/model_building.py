import os
import yaml
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# -------------------- Logging --------------------
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------- Utilities --------------------
def get_root_directory() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug(f'Parameters loaded from {params_path}')
    return params

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug(f'Data loaded from {file_path}, shape={df.shape}')
    return df

def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.debug(f'Model saved to {path}')

def save_vectorizer(vectorizer, path: str):
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.debug(f'Vectorizer saved to {path}')

def plot_confusion_matrix(cm, path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(path)
    plt.close()
    logger.debug(f'Confusion matrix saved to {path}')

# -------------------- TF-IDF --------------------
def apply_tfidf(df, max_features: int, ngram_range: tuple):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df['Content'])
    y = df['Label'].values
    logger.debug(f"TF-IDF applied. Shape: {X.shape}")
    return X, y, vectorizer

# -------------------- LightGBM Training --------------------
def train_lgbm(X, y, learning_rate: float, max_depth: int, n_estimators: int):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    logger.debug(f"SMOTE applied: original={len(y)}, resampled={len(y_res)}")

    n_classes = len(np.unique(y))
    logger.debug(f"number of unique classes: {n_classes}")

    if n_classes == 2:
        model = lgb.LGBMClassifier(
            objective='binary',
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42
        )
    else:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=n_classes,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42
        )

    model.fit(X_res, y_res)
    logger.debug("LightGBM training complete")
    return model

# -------------------- Evaluation --------------------
def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, f"{title}_confusion_matrix.png", title)
    logger.debug(f"{title} Accuracy: {acc}")
    logger.debug(f"{title} Classification Report:\n{report}")
    return acc, report

# -------------------- Main --------------------
def main():
    root_dir = get_root_directory()
    params = load_params(os.path.join(root_dir, 'params.yaml'))

    max_features = params['model_building']['max_features']
    ngram_range = tuple(params['model_building']['ngram_range'])
    lgb_estimators = params['model_building']['lgb_n_estimators']
    lgb_max_depth = params['model_building']['lgb_max_depth']
    lgb_lr = params['model_building']['lgb_learning_rate']

    # Load preprocessed data
    df = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

    # TF-IDF
    X, y, vectorizer = apply_tfidf(df, max_features, ngram_range)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train LightGBM
    lgb_model = train_lgbm(X_train, y_train, lgb_lr, lgb_max_depth, lgb_estimators)
    lgb_acc, lgb_report = evaluate_model(lgb_model, X_test, y_test, title="LightGBM")

    # Save artifacts
    save_model(lgb_model, os.path.join(root_dir, 'lgbm_model.pkl'))
    save_vectorizer(vectorizer, os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

    print("LightGBM Accuracy:", lgb_acc)

if __name__ == '__main__':
    main()
