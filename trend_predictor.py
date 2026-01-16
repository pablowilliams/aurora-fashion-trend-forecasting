"""Trend score prediction model"""
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class TrendPredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        return self.model.feature_importances_
