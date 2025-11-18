import pickle

# Load model
with open("lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_content(content):
    content = content.lower().strip()
    content = ' '.join([lemmatizer.lemmatize(word) for word in content.split()])
    return content

for comment in ["I hate this", "I love this"]:
    processed = preprocess_content(comment)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)
    print(comment, pred)

X = vectorizer.transform(preprocessed)

predictions = model.predict(X)
print(list(predictions))

probs = model.predict_proba(X)  # Only works for models with probability support
print(probs)
