import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

class OSCERiskPredictor:
    def __init__(self):
        self.model = XGBClassifier()
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return {"status": "trained"}

    def predict_risk(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
        self.is_trained = True
