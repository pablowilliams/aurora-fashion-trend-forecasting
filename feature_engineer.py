"""Feature engineering for fashion trend prediction"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def create_features(self, df):
        df = df.copy()
        df['engagement_score'] = np.log1p(df['page_views']) + np.log1p(df['wishlist_adds']) * 1.5
        df['conversion_rate'] = df['units_sold'] / (df['page_views'] + 1)
        df['price_ratio'] = df['current_price'] / df['original_price']
        df['review_density'] = df['num_reviews'] / (df['units_sold'] + 1)
        return df
