import os
import yaml
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json

# -------------------- Logging --------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------- Utilities --------------------
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

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.debug(f'Model loaded from {model_path}')
    return model

def load_vectorizer(vectorizer_path: str):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.debug(f'Vectorizer loaded from {vectorizer_path}')
    return vectorizer

def plot_confusion_matrix(cm, path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(path)
    plt.close()
    logger.debug(f'Confusion matrix saved to {path}')

def save_experiment_info(run_id: str, acc: float, file_path: str):
    info = {"run_id": run_id, "accuracy": acc}
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=4)
    logger.debug(f'Experiment info saved to {file_path}')

# -------------------- Main --------------------
def main():
    mlflow.set_tracking_uri("http://ec2-44-222-128-31.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("dvc-lightgbm-evaluation")

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log model parameters
            for key, value in params['model_building'].items():
                mlflow.log_param(key, value)

            # Load artifacts
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data
            test_df = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test = vectorizer.transform(test_df['Content'])
            y_test = test_df['Label'].values

            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"{label}_precision": metrics['precision'],
                        f"{label}_recall": metrics['recall'],
                        f"{label}_f1": metrics['f1-score']
                    })

            # Confusion matrix
            cm_path = os.path.join(root_dir, "confusion_matrix_test.png")
            plot_confusion_matrix(cm, cm_path, "LightGBM Test Confusion Matrix")
            mlflow.log_artifact(cm_path)

            # Save experiment info
            save_experiment_info(run.info.run_id, acc, os.path.join(root_dir, "experiment_info.json"))

            print("LightGBM Test Accuracy:", acc)

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
