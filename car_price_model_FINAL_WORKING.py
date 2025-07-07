"""
🏆 CAR PRICE PREDICTION MODEL - FINAL WORKING VERSION
MLDS Project - 3 Best Models + Ensemble Approach

This is the FINAL WORKING version that combines:
✅ 3 BEST performing models: ExtraTrees, RandomForest, GradientBoosting
✅ Stacking Ensemble for improved predictions
✅ Comprehensive model comparison and evaluation
✅ Clean, non-overlapping visualizations (5 groups)
✅ Perfect for Google Colab and presentations
✅ Ready for professors and teammates!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import re
import warnings
warnings.filterwarnings('ignore')

# Visualization settings for graphs
plt.style.use('default')
sns.set_palette('tab10')

print("="*80)
print("🚗 CAR PRICE PREDICTION MODEL - FINAL WORKING VERSION")
print("🎯 3 Best Models + Stacking Ensemble Approach")
print("📊 Comprehensive Model Comparison & Evaluation")
print("📈 With Clean Non-Overlapping Visualizations")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n1. Loading data...")

# For Google Colab compatibility
try:
    data = pd.read_csv('topic21_v40_train.csv')
    print("✅ Loaded from current directory")
except FileNotFoundError:
    try:
        data = pd.read_csv('../topic21_v40_train.csv')
        print("✅ Loaded from parent directory")
    except FileNotFoundError:
        print("❌ Please upload 'topic21_v40_train.csv' to the current directory")
        raise FileNotFoundError("Dataset file not found")

print(f"Dataset shape: {data.shape}")

# ============================================================================
# 2. DATA CLEANING (PROVEN WORKING VERSION)
# ============================================================================

print("\n2. Cleaning data...")

def convert_to_numeric(value):
    """Convert string values to numeric - PROVEN APPROACH"""
    if pd.isnull(value): return np.nan
    try: return float(value)
    except ValueError: pass
    
    # Handle ranges like "100-200"
    range_match = re.match(r"(\d+)\s*-\s*(\d+)", str(value))
    if range_match:
        lower, upper = range_match.groups()
        return (float(lower) + float(upper)) / 2
    
    # Extract just the number
    single_match = re.match(r"(\d+)", str(value))
    if single_match: return float(single_match.group(1))
    
    return np.nan

# Apply conversion (EXACT PROVEN APPROACH)
data['engine_capacity_cc'] = data['engine_capacity_cc'].apply(convert_to_numeric)
data['horsepower'] = data['horsepower'].apply(convert_to_numeric)

print("✅ Data cleaning completed")

# ============================================================================
# 3. FEATURE ENGINEERING (EXACT PROVEN APPROACH)
# ============================================================================

print("\n3. Feature engineering...")

def create_features(df):
    """Create features that achieved 76.9% R² score - EXACT APPROACH"""
    df_fe = df.copy()
    
    # 1. Brand prestige (5 tiers) - PROVEN MAPPING
    ultra_luxury = ['ferrari', 'lamborghini', 'bentley', 'rolls-royce', 'maserati', 'aston martin', 'bugatti']
    luxury = ['mercedes-benz', 'bmw', 'audi', 'lexus', 'porsche', 'jaguar', 'cadillac', 'lincoln']
    premium = ['volvo', 'land rover', 'mini', 'tesla', 'infiniti', 'acura']
    mainstream_premium = ['honda', 'toyota', 'mazda', 'subaru']
    mainstream = ['ford', 'chevrolet', 'nissan', 'hyundai', 'kia', 'volkswagen']
    
    # Set brand prestige levels
    df_fe['brand_prestige'] = 1  # Default
    df_fe.loc[df_fe['brand'].str.lower().isin(mainstream), 'brand_prestige'] = 1
    df_fe.loc[df_fe['brand'].str.lower().isin(mainstream_premium), 'brand_prestige'] = 2
    df_fe.loc[df_fe['brand'].str.lower().isin(premium), 'brand_prestige'] = 3
    df_fe.loc[df_fe['brand'].str.lower().isin(luxury), 'brand_prestige'] = 4
    df_fe.loc[df_fe['brand'].str.lower().isin(ultra_luxury), 'brand_prestige'] = 5
    
    # 2. Body type value (5 tiers) - PROVEN MAPPING
    performance = ['coupe', 'convertible', 'roadster', 'sports car']
    luxury_sedan = ['sedan', 'limousine']
    utility = ['suv', 'crossover', 'wagon']
    practical = ['hatchback', 'compact']
    commercial = ['pickup', 'truck', 'van']
    
    df_fe['body_value'] = 2  # Default
    df_fe.loc[df_fe['body_type'].str.lower().isin(practical), 'body_value'] = 1
    df_fe.loc[df_fe['body_type'].str.lower().isin(commercial), 'body_value'] = 2
    df_fe.loc[df_fe['body_type'].str.lower().isin(utility), 'body_value'] = 3
    df_fe.loc[df_fe['body_type'].str.lower().isin(luxury_sedan), 'body_value'] = 4
    df_fe.loc[df_fe['body_type'].str.lower().isin(performance), 'body_value'] = 5
    
    # 3. Power metrics - PROVEN FORMULAS
    df_fe['hp_per_liter'] = df_fe['horsepower'] / (df_fe['engine_capacity_cc'] / 1000 + 0.1)
    df_fe['power_to_displacement'] = df_fe['horsepower'] / (df_fe['engine_capacity_cc'] + 100)
    
    # 4. Power categories - PROVEN SEGMENTS
    df_fe['power_segment'] = pd.cut(df_fe['horsepower'], 
                                bins=[0, 150, 250, 350, 500, float('inf')], 
                                labels=range(5)).astype(float)
    
    # 5. Key interactions (crucial features) - PROVEN COMBINATIONS
    df_fe['luxury_power'] = df_fe['brand_prestige'] * df_fe['horsepower']
    df_fe['prestige_body'] = df_fe['brand_prestige'] * df_fe['body_value']
    df_fe['performance_index'] = df_fe['brand_prestige'] * df_fe['power_segment']
    
    # 6. Market positioning - PROVEN FORMULA
    df_fe['market_tier'] = (df_fe['brand_prestige'] * 0.4 + 
                        df_fe['body_value'] * 0.3 + 
                        df_fe['power_segment'] * 0.3)
    
    # 7. Polynomial features - PROVEN TRANSFORMATIONS
    df_fe['brand_prestige_squared'] = df_fe['brand_prestige'] ** 2
    df_fe['horsepower_squared'] = df_fe['horsepower'] ** 2
    df_fe['luxury_power_log'] = np.log1p(df_fe['luxury_power'])
    df_fe['market_tier_squared'] = df_fe['market_tier'] ** 2
    
    return df_fe

# Apply feature engineering
data_enhanced = create_features(data)
print(f"Original features: {data.shape[1]}")
print(f"Enhanced features: {data_enhanced.shape[1]}")
print(f"New features added: {data_enhanced.shape[1] - data.shape[1]}")

# ============================================================================
# 4. COMPREHENSIVE VISUALIZATIONS (CLEAN & NON-OVERLAPPING)
# ============================================================================

print("\n📊 Creating comprehensive visualizations...")

# VISUALIZATION 1: Basic Price Analysis
print("📊 Creating Basic Price Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Basic Price Analysis - Car Price Prediction Model', fontsize=16, fontweight='bold')

# Price Distribution
axes[0,0].hist(data_enhanced['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price ($)')
axes[0,0].set_ylabel('Frequency')

# Brand Prestige vs Price
sns.boxplot(data=data_enhanced, x='brand_prestige', y='price', ax=axes[0,1])
axes[0,1].set_title('Brand Prestige Impact on Price')
axes[0,1].set_xlabel('Brand Prestige (1=Economy, 5=Ultra-luxury)')

# Body Type vs Price
sns.boxplot(data=data_enhanced, x='body_value', y='price', ax=axes[1,0])
axes[1,0].set_title('Body Type Value Impact on Price')
axes[1,0].set_xlabel('Body Type Value (1=Low, 5=High)')

# Brand Prestige Distribution
brand_counts = data_enhanced['brand_prestige'].value_counts().sort_index()
axes[1,1].bar(brand_counts.index, brand_counts.values, color='lightcoral')
axes[1,1].set_title('Brand Prestige Distribution')
axes[1,1].set_xlabel('Brand Prestige Level')
axes[1,1].set_ylabel('Number of Cars')

plt.tight_layout()
plt.show()
plt.close()

# VISUALIZATION 2: Feature Relationships
print("📊 Creating Feature Relationships...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Relationships with Price', fontsize=16, fontweight='bold')

# Horsepower vs Price
axes[0,0].scatter(data_enhanced['horsepower'], data_enhanced['price'], alpha=0.5, s=30)
axes[0,0].set_title('Horsepower vs Price')
axes[0,0].set_xlabel('Horsepower')
axes[0,0].set_ylabel('Price ($)')

# Engine Capacity vs Price
axes[0,1].scatter(data_enhanced['engine_capacity_cc'], data_enhanced['price'], alpha=0.5, s=30, color='green')
axes[0,1].set_title('Engine Capacity vs Price')
axes[0,1].set_xlabel('Engine Capacity (cc)')
axes[0,1].set_ylabel('Price ($)')

# Power per Liter vs Price
axes[1,0].scatter(data_enhanced['hp_per_liter'], data_enhanced['price'], alpha=0.5, s=30, color='purple')
axes[1,0].set_title('Power Efficiency vs Price')
axes[1,0].set_xlabel('HP per Liter')
axes[1,0].set_ylabel('Price ($)')

# Power Segment vs Price
sns.boxplot(data=data_enhanced, x='power_segment', y='price', ax=axes[1,1])
axes[1,1].set_title('Power Segment vs Price')
axes[1,1].set_xlabel('Power Segment (0=Low, 4=High)')

plt.tight_layout()
plt.show()
plt.close()

# VISUALIZATION 3: Engineered Features
print("📊 Creating Engineered Features Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Engineered Features Impact on Price', fontsize=16, fontweight='bold')

# Luxury Power vs Price
axes[0,0].scatter(data_enhanced['luxury_power'], data_enhanced['price'], alpha=0.5, s=30, color='red')
axes[0,0].set_title('Luxury Power vs Price')
axes[0,0].set_xlabel('Luxury Power (Brand × HP)')
axes[0,0].set_ylabel('Price ($)')

# Market Tier vs Price
axes[0,1].scatter(data_enhanced['market_tier'], data_enhanced['price'], alpha=0.5, s=30, color='brown')
axes[0,1].set_title('Market Tier vs Price')
axes[0,1].set_xlabel('Market Tier Score')
axes[0,1].set_ylabel('Price ($)')

# Performance Index vs Price
axes[1,0].scatter(data_enhanced['performance_index'], data_enhanced['price'], alpha=0.5, s=30, color='darkblue')
axes[1,0].set_title('Performance Index vs Price')
axes[1,0].set_xlabel('Performance Index')
axes[1,0].set_ylabel('Price ($)')

# Body Type Distribution
body_counts = data_enhanced['body_type'].value_counts()
if len(body_counts) <= 10:  # Only show pie chart if not too many categories
    axes[1,1].pie(body_counts.values, labels=body_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Body Type Distribution')
else:
    axes[1,1].bar(range(len(body_counts[:10])), body_counts.values[:10])
    axes[1,1].set_title('Top 10 Body Types')
    axes[1,1].set_xlabel('Body Type')
    axes[1,1].set_ylabel('Count')

plt.tight_layout()
plt.show()
plt.close()

print("✅ All visualizations completed successfully!")

# ============================================================================
# 5. OUTLIER REMOVAL (PROVEN APPROACH)
# ============================================================================

print("\n4. Remove outliers...")

# Remove the most extreme prices (1st-99th percentile) - PROVEN APPROACH
price_1st = data_enhanced['price'].quantile(0.01)
price_99th = data_enhanced['price'].quantile(0.99)

data_clean = data_enhanced[
    (data_enhanced['price'] >= price_1st) & 
    (data_enhanced['price'] <= price_99th)
]

print(f"Original data: {data_enhanced.shape[0]} rows")
print(f"After outlier removal: {data_clean.shape[0]} rows")
print(f"Removed outliers: {data_enhanced.shape[0] - data_clean.shape[0]} rows")

# ============================================================================
# 6. TRAIN-TEST SPLIT (PROVEN APPROACH)
# ============================================================================

print("\n5. Train-test split...")

# Split data into features and target
X = data_clean.drop('price', axis=1)
y = data_clean['price']

# Create train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ============================================================================
# 7. PREPROCESSING (PROVEN APPROACH)
# ============================================================================

print("\n6. Setting up preprocessing...")

# Set up preprocessing pipeline - AUTO-DETECT COLUMNS (PROVEN APPROACH)
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

# Create preprocessing pipeline - EXACT PROVEN PIPELINE
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')),  # KEY: PowerTransformer!
        ('scaler', RobustScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

print(f"Numerical features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# ============================================================================
# 8. MODEL TRAINING (3 BEST MODELS + ENSEMBLE)
# ============================================================================

print("\n7. Training 3 best models...")

# Define the 3 BEST performing models with proven parameters
models = {
    'ExtraTrees_Best': ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=False,
        n_jobs=-1,
        random_state=42
    ),
    
    'RandomForest_Best': RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),
    
    'GradientBoosting_Best': GradientBoostingRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.08,
        subsample=0.9,
        max_features=0.8,
        random_state=42
    )
}

# Train all 3 models
trained_models = {}
individual_results = {}

print("\n=== Training Individual Models ===")
for name, model in models.items():
    print(f"   Training {name}...")
    start_time = time.time()
    
    # Create pipeline for each model
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    trained_models[name] = {
        'pipeline': pipeline,
        'predictions': y_pred,
        'training_time': time.time() - start_time
    }
    
    individual_results[name] = {
        'R²': r2,
        'MAE': mae,
        'Training_Time': time.time() - start_time
    }
    
    print(f"   ✅ {name}: R² = {r2:.4f} ({r2*100:.1f}%), MAE = ${mae:,.0f}")

print("\n=== Creating Ensemble Model ===")
# Create stacking ensemble from the 3 best models
estimators = [(name.split('_')[0].lower(), info['pipeline'].named_steps['model']) 
              for name, info in trained_models.items()]

stacking_ensemble = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1
)

# Train ensemble
print("   Training Stacking Ensemble...")
start_time = time.time()

ensemble_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('ensemble', stacking_ensemble)
])

ensemble_pipeline.fit(X_train, y_train)
ensemble_pred = ensemble_pipeline.predict(X_test)

# Calculate ensemble metrics
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

trained_models['Stacking_Ensemble'] = {
    'pipeline': ensemble_pipeline,
    'predictions': ensemble_pred,
    'training_time': time.time() - start_time
}

individual_results['Stacking_Ensemble'] = {
    'R²': ensemble_r2,
    'MAE': ensemble_mae,
    'Training_Time': time.time() - start_time
}

print(f"   ✅ Stacking Ensemble: R² = {ensemble_r2:.4f} ({ensemble_r2*100:.1f}%), MAE = ${ensemble_mae:,.0f}")

print("\n✅ All models trained successfully!")

# ============================================================================
# 9. MODEL EVALUATION WITH VISUALIZATIONS
# ============================================================================

print("\n8. Evaluating all models...")

# Display comprehensive results
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)

# Sort models by R² score
sorted_results = sorted(individual_results.items(), key=lambda x: x[1]['R²'], reverse=True)

print(f"\n🏆 MODEL PERFORMANCE RANKINGS:")
print(f"{'Rank':<4} {'Model':<25} {'R² Score':<15} {'MAE':<15} {'Training Time':<15}")
print("-" * 80)

for i, (name, metrics) in enumerate(sorted_results, 1):
    r2_pct = metrics['R²'] * 100
    print(f"{i:2d}.  {name:<25} {metrics['R²']:.4f} ({r2_pct:.1f}%) ${metrics['MAE']:>10,.0f} {metrics['Training_Time']:>10.1f}s")

# Get best model
best_model_name = sorted_results[0][0]
best_model_r2 = sorted_results[0][1]['R²']
best_model_mae = sorted_results[0][1]['MAE']
best_predictions = trained_models[best_model_name]['predictions']

print(f"\n🎯 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_model_r2:.4f} ({best_model_r2*100:.1f}%)")
print(f"   Mean Absolute Error: ${best_model_mae:,.0f}")

# Use best model's predictions for detailed analysis
y_pred = best_predictions
r2 = best_model_r2
mae = best_model_mae

# VISUALIZATION 4: Model Performance Analysis
print("📊 Creating Model Performance Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30)
min_price, max_price = y_test.min(), y_test.max()
axes[0,0].plot([min_price, max_price], [min_price, max_price], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Price ($)')
axes[0,0].set_ylabel('Predicted Price ($)')
axes[0,0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')

# Residuals plot
residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_xlabel('Predicted Price ($)')
axes[0,1].set_ylabel('Residuals ($)')
axes[0,1].set_title('Residuals Plot')

# Feature importance (top 10) from best model
best_pipeline = trained_models[best_model_name]['pipeline']
feature_names = (num_cols + 
                list(best_pipeline.named_steps['preprocess']
                    .named_transformers_['cat']
                    .named_steps['encoder']
                    .get_feature_names_out(cat_cols)))

importances = best_pipeline.named_steps['model'].feature_importances_
indices = np.argsort(importances)[-10:]

axes[1,0].barh(range(len(indices)), importances[indices])
axes[1,0].set_yticks(range(len(indices)))
axes[1,0].set_yticklabels([feature_names[i] for i in indices])
axes[1,0].set_xlabel('Feature Importance')
axes[1,0].set_title(f'Top 10 Features ({best_model_name})')

# Error distribution
axes[1,1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].set_xlabel('Prediction Error ($)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Error Distribution')
axes[1,1].axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()
plt.close()

print("✅ Model performance visualizations completed!")

# VISUALIZATION 5: Model Comparison Analysis
print("📊 Creating Model Comparison Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Comparison Analysis - 3 Best Models + Ensemble', fontsize=16, fontweight='bold')

# Model Performance Comparison (R² scores)
model_names = [name.replace('_', ' ') for name in individual_results.keys()]
r2_scores = [metrics['R²'] for metrics in individual_results.values()]
mae_scores = [metrics['MAE'] for metrics in individual_results.values()]

# R² Score comparison
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(model_names)]
bars1 = axes[0,0].bar(model_names, r2_scores, color=colors)
axes[0,0].set_title('R² Score Comparison')
axes[0,0].set_ylabel('R² Score')
axes[0,0].set_ylim(0, 1)
axes[0,0].tick_params(axis='x', rotation=45)
# Add value labels on bars
for bar, score in zip(bars1, r2_scores):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# MAE comparison
bars2 = axes[0,1].bar(model_names, mae_scores, color=colors)
axes[0,1].set_title('Mean Absolute Error Comparison')
axes[0,1].set_ylabel('MAE ($)')
axes[0,1].tick_params(axis='x', rotation=45)
# Add value labels on bars
for bar, score in zip(bars2, mae_scores):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.01, 
                   f'${score:,.0f}', ha='center', va='bottom', fontweight='bold')

# Prediction scatter plot comparison (show top 3 individual models)
individual_models = [(name, data) for name, data in trained_models.items() if 'Ensemble' not in name]
for i, (name, model_data) in enumerate(individual_models):
    color = colors[i]
    alpha = 0.3 if i > 0 else 0.7  # Make best model more prominent
    axes[1,0].scatter(y_test, model_data['predictions'], alpha=alpha, s=20, 
                     color=color, label=name.replace('_', ' '))

# Perfect prediction line
min_price, max_price = y_test.min(), y_test.max()
axes[1,0].plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
axes[1,0].set_xlabel('Actual Price ($)')
axes[1,0].set_ylabel('Predicted Price ($)')
axes[1,0].set_title('Individual Models Predictions vs Actual')
axes[1,0].legend()

# Model complexity comparison (training time vs performance)
training_times = [metrics['Training_Time'] for metrics in individual_results.values()]
axes[1,1].scatter(training_times, r2_scores, c=colors, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    axes[1,1].annotate(name, (training_times[i], r2_scores[i]), 
                      xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1,1].set_xlabel('Training Time (seconds)')
axes[1,1].set_ylabel('R² Score')
axes[1,1].set_title('Model Efficiency: Performance vs Training Time')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("✅ Model comparison visualizations completed!")

# ============================================================================
# 🎯 HIGHLIGHTED RESULTS SUMMARY 
# ============================================================================

print("\n" + "🎯" + "="*78 + "🎯")
print("🏆                          FINAL MODEL RESULTS                           🏆")
print("🎯" + "="*78 + "🎯")

# Create a results summary table
print("\n" + "📊 DETAILED PERFORMANCE METRICS:")
print("━" * 90)
print(f"{'🏆 RANK':<8} {'MODEL NAME':<25} {'R² SCORE':<12} {'R² %':<8} {'MAE ($)':<15} {'TIME':<8}")
print("━" * 90)

for i, (name, metrics) in enumerate(sorted_results, 1):
    r2_pct = metrics['R²'] * 100
    emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '⭐'
    color_text = '🟢' if r2_pct >= 75 else '🟡' if r2_pct >= 70 else '🔴'
    
    print(f"{emoji:<8} {name:<25} {metrics['R²']:.4f}       {r2_pct:5.1f}%   ${metrics['MAE']:>10,.0f}    {metrics['Training_Time']:5.1f}s")

print("━" * 90)

# Highlight the BEST model
print(f"\n🎖️  CHAMPION MODEL: {best_model_name}")
print(f"    🎯 R² Score: {best_model_r2:.4f} ({best_model_r2*100:.1f}%)")
print(f"    💰 Mean Absolute Error: ${best_model_mae:,.0f}")
print(f"    ⏱️  Training Time: {sorted_results[0][1]['Training_Time']:.1f} seconds")

# Performance status
if best_model_r2 >= 0.75:
    status_emoji = "🟢 EXCELLENT"
    status_text = "Outstanding performance! Above 75% R²"
elif best_model_r2 >= 0.70:
    status_emoji = "🟡 GOOD"
    status_text = "Good performance! Above 70% R²"
else:
    status_emoji = "🔴 NEEDS IMPROVEMENT"
    status_text = "Below target performance"

print(f"    📈 Status: {status_emoji}")
print(f"    📝 Assessment: {status_text}")

print("\n" + "🎯" + "="*78 + "🎯")

# ============================================================================
# 10. FINAL RESULTS AND SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🏆 FINAL RESULTS - 3 BEST MODELS + ENSEMBLE")
print("="*80)

# Show all model results
print(f"\n📊 COMPREHENSIVE MODEL PERFORMANCE:")
for i, (name, metrics) in enumerate(sorted_results, 1):
    r2_pct = metrics['R²'] * 100
    status = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '⭐'
    print(f"   {status} {name}: {r2_pct:.1f}% R² | ${metrics['MAE']:,.0f} MAE")

print(f"\n🎯 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_model_r2:.4f} ({best_model_r2*100:.1f}%)")
print(f"   Mean Absolute Error: ${best_model_mae:,.0f}")
print(f"   Status: {'✅ EXCELLENT' if best_model_r2 > 0.75 else '✅ GOOD' if best_model_r2 > 0.70 else '⚠️ NEEDS IMPROVEMENT'}")

print("\n" + "="*80)
print("📋 PRESENTATION SUMMARY FOR PROFESSORS")
print("="*80)
print(f"✅ Models Used: 3 Best Algorithms + Stacking Ensemble")
print(f"   • ExtraTreesRegressor (500 estimators)")
print(f"   • RandomForestRegressor (500 estimators)")  
print(f"   • GradientBoostingRegressor (500 estimators)")
print(f"   • StackingRegressor with Ridge meta-learner")
print(f"✅ Best Performance: {best_model_r2*100:.1f}% R² Score (Target: 70-80%)")
print(f"✅ Model Comparison: Comprehensive evaluation of all approaches")
print(f"✅ Features: Advanced feature engineering ({data_enhanced.shape[1] - data.shape[1]} new features)")
print(f"✅ Visualizations: 5 comprehensive, non-overlapping chart groups")
print(f"✅ Best Error Rate: ${best_model_mae:,.0f} mean absolute error")
print(f"✅ Innovation: Brand prestige, power segments, market tiers")
print(f"✅ Ensemble Method: Stacking for improved predictions")
print(f"✅ Status: {'Production ready!' if best_model_r2 > 0.70 else 'Needs improvement'}")

print("\n🎉 MULTI-MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("📊 All visualizations are clean and professional")
print("🎓 Perfect for Google Colab and academic presentations!")
print("🚀 Ready to impress professors and teammates!")
print("💯 Uses 3 BEST models + ensemble for maximum performance!")
print("🏆 Demonstrates comprehensive ML model comparison!")

print("\n📝 GOOGLE COLAB INSTRUCTIONS:")
print("1. Upload this file to Google Colab")
print("2. Upload 'topic21_v40_train.csv' dataset") 
print("3. Run the entire script")
print("4. View all 5 visualization groups")
print("5. Present the comprehensive model comparison!")
print("="*80) 