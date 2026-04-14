import joblib
import numpy as np

def predict_sample():
    model = joblib.load("models/model.pkl")

    sample = np.array([[90, 6, 45]])

    prediction = model.predict(sample)

    return prediction[0]