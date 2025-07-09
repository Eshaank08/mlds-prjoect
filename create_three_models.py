#!/usr/bin/env python3
"""
Create Three Simple Car Price Models
===================================
Creates 3 different models with NO feature engineering - just raw data.
"""

import pandas as pd
import numpy as np
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_pipeline(model):
    """Create a simple preprocessing pipeline for any model."""
    
    # Define feature types based on data types
    numeric_features = ['0', '1', '2', '3', '4']  # Anonymous numeric columns
    categorical_features = ['brand', 'model', 'trim', 'body_type', 'fuel_type', 
                          'transmission_type', 'engine_capacity_cc', 'horsepower',
                          'exterior_color', 'interior_color', 'warranty', 'city', 'seller_type']
    
    # Simple preprocessing
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def main():
    print("🚗 Creating Three Simple Car Price Models")
    print("=" * 50)
    print("No feature engineering - just raw data!")
    
    # Load data
    print("\n1. Loading dataset...")
    data = pd.read_csv('topic21_v40_train.csv')
    print(f"   Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Basic cleaning - only remove rows with missing price
    data_clean = data.dropna(subset=['price'])
    print(f"   Cleaned: {data_clean.shape[0]} rows")
    
    # Prepare features and target
    X = data_clean.drop('price', axis=1)
    y = data_clean['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define three different models - all good performers
    models = [
        ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    
    print(f"\n2. Training {len(models)} models...")
    
    for i, (name, model) in enumerate(models, 1):
        print(f"\n   Model {i}: {name}")
        
        # Create pipeline
        pipeline = create_pipeline(model)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"      R² Score: {r2:.4f} ({r2*100:.1f}%)")
        print(f"      MAE: ${mae:,.0f}")
        
        # Save model
        filename = f"model_{i}.skops"
        sio.dump(pipeline, filename)
        print(f"      ✅ Saved: {filename}")
    
    print(f"\n🎉 All three models created successfully!")
    print(f"   Files: model_1.skops, model_2.skops, model_3.skops")

if __name__ == "__main__":
    main() 