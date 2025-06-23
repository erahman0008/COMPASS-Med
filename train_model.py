import pandas as pd
from src.models.risk_predictor import OSCERiskPredictor

if __name__ == '__main__':
    df = pd.read_csv('data/processed/training_features.csv')
    X = df.drop(columns='target')
    y = df['target']
    model = OSCERiskPredictor()
    model.train(X, y)
    model.save_model('models/trained_model.pkl')
