"""Data loading and preprocessing for fashion trend prediction"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.encoders = {}
        
    def load_data(self):
        df = pd.read_csv(self.filepath)
        return self._preprocess(df)
    
    def _preprocess(self, df):
        categorical_cols = ['category', 'subcategory', 'brand', 'color', 'pattern', 'material', 'season', 'gender']
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
        return df
