#!/usr/bin/env python3
"""
Test Three Skops Models
======================
Basic script to load and test the three model files.
"""

import pandas as pd
import numpy as np
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def test_model(model_file):
    """Test a single skops model file."""
    print(f"\n🔍 Testing: {model_file}")
    print("-" * 30)
    
    try:
        # Load data
        data = pd.read_csv('topic21_v40_train.csv')
        X = data.drop('price', axis=1)
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load model
        untrusted_types = sio.get_untrusted_types(file=model_file)
        model = sio.load(model_file, trusted=untrusted_types)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   ✅ R² Score: {r2:.4f} ({r2*100:.1f}%)")
        print(f"   💰 MAE: ${mae:,.0f}")
        
        return r2, mae
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return 0.0, float('inf')

def main():
    print("🚗 Testing Three Car Price Models")
    print("=" * 40)
    
    # Test all three models
    model_files = ['model_1.skops', 'model_2.skops', 'model_3.skops']
    model_names = ['Extra Trees', 'Random Forest', 'Gradient Boosting']
    
    results = []
    
    for model_file, name in zip(model_files, model_names):
        print(f"\n📊 {name}")
        r2, mae = test_model(model_file)
        results.append((name, r2, mae))
    
    # Summary
    print(f"\n🏆 RESULTS SUMMARY")
    print("=" * 40)
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by R²
    
    for i, (name, r2, mae) in enumerate(results, 1):
        print(f"{i}. {name}")
        print(f"   R²: {r2*100:.1f}% | MAE: ${mae:,.0f}")
    
    best_model = results[0]
    print(f"\n🥇 Best: {best_model[0]} ({best_model[1]*100:.1f}% R²)")

if __name__ == "__main__":
    main() 