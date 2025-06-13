from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load tokenizer (assume still in pickle format)
tokenizer = joblib.load("tokenizer.pkl")

# Load models using joblib
lgbm_model = joblib.load("lgbm_model.pkl")
catboost_model = joblib.load("catboost_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Model dictionary with descriptions
models = {
    "LGBM": (lgbm_model, "LGBM is a gradient boosting framework optimized for performance and scalability."),
    "CatBoost": (catboost_model, "CatBoost handles categorical features and combats overfitting via ordered boosting."),
    "XGBoost": (xgb_model, "XGBoost is an optimized gradient boosting library that's highly accurate and fast."),
    "Random Forest": (rf_model, "Random Forest uses multiple decision trees and is robust to overfitting.")
}

# Flask App setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_info = ""
    selected_model = None

    if request.method == "POST":
        input_text = request.form["text"]
        input_text = preprocess_text(input_text)

        selected_model = request.form["model"]
        model, model_info = models[selected_model]

        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=546)

        # Standardize (per sample)
        scaler = StandardScaler()
        scaled_input = scaler.fit_transform(padded)

        # Predict
        result = model.predict(scaled_input)
        if hasattr(result, 'toarray'):
            result = result.toarray()
        label = int(np.round(result[0])) if result.ndim > 1 else int(np.round(result))
        prediction = "Sarcastic" if label == 1 else "Not Sarcastic"

    return render_template("index.html", prediction=prediction, model_info=model_info, selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
