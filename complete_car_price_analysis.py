# %% [markdown]
# # Complete Car Price Prediction Analysis
# ## Raw Data vs Feature Engineering Comparison
# 
# This notebook compares different machine learning models with and without feature engineering
# to predict car prices and visualize the improvements.

# %%
# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor, 
                            GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("🚗 Car Price Prediction Analysis")
print("=" * 50)

# %%
# Load and explore the data
print("📊 Loading and exploring data...")
data = pd.read_csv('topic21_v40_train.csv')

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"\nBasic statistics:")
print(data.describe())

# %%
# Data Visualization - Raw Data Analysis
print("📈 Creating visualizations for raw data...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Car Price Analysis - Raw Data Exploration', fontsize=16, fontweight='bold')

# Price distribution
axes[0,0].hist(data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Price Distribution', fontweight='bold')
axes[0,0].set_xlabel('Price ($)')
axes[0,0].set_ylabel('Frequency')

# Price vs Brand (top 10 brands)
top_brands = data['brand'].value_counts().head(10).index
brand_prices = data[data['brand'].isin(top_brands)].groupby('brand')['price'].mean().sort_values(ascending=False)
axes[0,1].bar(range(len(brand_prices)), brand_prices.values, color='lightcoral')
axes[0,1].set_title('Average Price by Brand (Top 10)', fontweight='bold')
axes[0,1].set_xlabel('Brand')
axes[0,1].set_ylabel('Average Price ($)')
axes[0,1].set_xticks(range(len(brand_prices)))
axes[0,1].set_xticklabels(brand_prices.index, rotation=45, ha='right')

# Price vs Body Type
body_prices = data.groupby('body_type')['price'].mean().sort_values(ascending=False)
axes[0,2].bar(range(len(body_prices)), body_prices.values, color='lightgreen')
axes[0,2].set_title('Average Price by Body Type', fontweight='bold')
axes[0,2].set_xlabel('Body Type')
axes[0,2].set_ylabel('Average Price ($)')
axes[0,2].set_xticks(range(len(body_prices)))
axes[0,2].set_xticklabels(body_prices.index, rotation=45, ha='right')

# Price vs Fuel Type
fuel_prices = data.groupby('fuel_type')['price'].mean().sort_values(ascending=False)
axes[1,0].bar(range(len(fuel_prices)), fuel_prices.values, color='gold')
axes[1,0].set_title('Average Price by Fuel Type', fontweight='bold')
axes[1,0].set_xlabel('Fuel Type')
axes[1,0].set_ylabel('Average Price ($)')
axes[1,0].set_xticks(range(len(fuel_prices)))
axes[1,0].set_xticklabels(fuel_prices.index, rotation=45, ha='right')

# Horsepower vs Price (scatter plot)
# Clean horsepower data for plotting
hp_clean = pd.to_numeric(data['horsepower'], errors='coerce')
valid_hp = ~hp_clean.isna()
axes[1,1].scatter(hp_clean[valid_hp], data.loc[valid_hp, 'price'], alpha=0.5, color='purple')
axes[1,1].set_title('Price vs Horsepower', fontweight='bold')
axes[1,1].set_xlabel('Horsepower')
axes[1,1].set_ylabel('Price ($)')

# Anonymous features correlation with price
anon_features = ['0', '1', '2', '3', '4']
correlations = [data[col].corr(data['price']) for col in anon_features if col in data.columns]
axes[1,2].bar(anon_features[:len(correlations)], correlations, color='orange')
axes[1,2].set_title('Anonymous Features vs Price Correlation', fontweight='bold')
axes[1,2].set_xlabel('Anonymous Feature')
axes[1,2].set_ylabel('Correlation with Price')

plt.tight_layout()
plt.show()

# %%
# Feature Engineering Function
def create_enhanced_features(df):
    """Create enhanced features for better model performance."""
    df_new = df.copy()
    
    # Clean numeric columns
    def clean_numeric(value):
        if pd.isnull(value):
            return np.nan
        try:
            return float(value)
        except:
            # Handle ranges and extract numbers
            import re
            num_match = re.search(r'(\d+\.?\d*)', str(value))
            if num_match:
                return float(num_match.group(1))
            return np.nan
    
    df_new['engine_capacity_cc'] = df_new['engine_capacity_cc'].apply(clean_numeric)
    df_new['horsepower'] = df_new['horsepower'].apply(clean_numeric)
    
    # Enhanced brand categorization
    luxury_brands = ['mercedes-benz', 'bmw', 'audi', 'lexus', 'porsche', 'jaguar']
    premium_brands = ['volvo', 'infiniti', 'cadillac', 'land rover', 'acura']
    
    df_new['brand_tier'] = 3  # default mainstream
    df_new.loc[df_new['brand'].str.lower().isin(premium_brands), 'brand_tier'] = 4
    df_new.loc[df_new['brand'].str.lower().isin(luxury_brands), 'brand_tier'] = 5
    
    # Performance categories
    df_new['power_category'] = pd.cut(df_new['horsepower'], 
                                      bins=[0, 150, 250, 400, 1000], 
                                      labels=[1, 2, 3, 4], 
                                      include_lowest=True).astype(float)
    
    # Engine size categories
    df_new['engine_category'] = pd.cut(df_new['engine_capacity_cc'], 
                                       bins=[0, 1500, 2500, 4000, 10000], 
                                       labels=[1, 2, 3, 4], 
                                       include_lowest=True).astype(float)
    
    # Power per liter
    df_new['power_per_liter'] = df_new['horsepower'] / (df_new['engine_capacity_cc'] / 1000 + 0.1)
    df_new['power_per_liter'] = df_new['power_per_liter'].clip(0, 300)
    
    # Interaction features
    df_new['brand_power_interaction'] = df_new['brand_tier'] * df_new['horsepower'] / 100
    
    # Anonymous feature combinations
    anon_cols = ['0', '1', '2', '3', '4']
    available_anon = [col for col in anon_cols if col in df_new.columns]
    
    if len(available_anon) >= 2:
        df_new['anon_sum'] = df_new[available_anon].sum(axis=1)
        df_new['anon_mean'] = df_new[available_anon].mean(axis=1)
        df_new['anon_std'] = df_new[available_anon].std(axis=1)
    
    # Binary indicators
    df_new['is_luxury'] = (df_new['brand_tier'] >= 5).astype(int)
    df_new['is_high_performance'] = (df_new['horsepower'] > 300).astype(int)
    
    return df_new

print("🔧 Creating enhanced features...")
data_enhanced = create_enhanced_features(data)
new_features = data_enhanced.shape[1] - data.shape[1]
print(f"Added {new_features} new features through feature engineering")

# %%
# Enhanced Data Visualization
print("📊 Creating visualizations for enhanced data...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Car Price Analysis - Enhanced Features', fontsize=16, fontweight='bold')

# Brand tier vs Price
brand_tier_prices = data_enhanced.groupby('brand_tier')['price'].mean()
axes[0,0].bar(brand_tier_prices.index, brand_tier_prices.values, color='steelblue')
axes[0,0].set_title('Average Price by Brand Tier', fontweight='bold')
axes[0,0].set_xlabel('Brand Tier (3=Mainstream, 4=Premium, 5=Luxury)')
axes[0,0].set_ylabel('Average Price ($)')

# Power category vs Price
power_cat_prices = data_enhanced.groupby('power_category')['price'].mean()
axes[0,1].bar(power_cat_prices.index, power_cat_prices.values, color='coral')
axes[0,1].set_title('Average Price by Power Category', fontweight='bold')
axes[0,1].set_xlabel('Power Category (1=Low, 4=High)')
axes[0,1].set_ylabel('Average Price ($)')

# Power per liter vs Price
valid_ppl = ~data_enhanced['power_per_liter'].isna()
axes[0,2].scatter(data_enhanced.loc[valid_ppl, 'power_per_liter'], 
                  data_enhanced.loc[valid_ppl, 'price'], alpha=0.5, color='green')
axes[0,2].set_title('Price vs Power per Liter', fontweight='bold')
axes[0,2].set_xlabel('Power per Liter')
axes[0,2].set_ylabel('Price ($)')

# Feature correlation heatmap
feature_cols = ['brand_tier', 'power_category', 'engine_category', 'power_per_liter', 
                'brand_power_interaction', 'anon_sum', 'anon_mean']
available_features = [col for col in feature_cols if col in data_enhanced.columns]
corr_data = data_enhanced[available_features + ['price']].corr()

im = axes[1,0].imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1,0].set_title('Feature Correlation Matrix', fontweight='bold')
axes[1,0].set_xticks(range(len(corr_data.columns)))
axes[1,0].set_yticks(range(len(corr_data.columns)))
axes[1,0].set_xticklabels(corr_data.columns, rotation=45, ha='right')
axes[1,0].set_yticklabels(corr_data.columns)

# Luxury vs Non-luxury price distribution
luxury_mask = data_enhanced['is_luxury'] == 1
axes[1,1].hist([data_enhanced[~luxury_mask]['price'], data_enhanced[luxury_mask]['price']], 
               bins=30, alpha=0.7, label=['Non-Luxury', 'Luxury'], color=['lightblue', 'gold'])
axes[1,1].set_title('Price Distribution: Luxury vs Non-Luxury', fontweight='bold')
axes[1,1].set_xlabel('Price ($)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

# High performance vs Regular performance
hp_mask = data_enhanced['is_high_performance'] == 1
axes[1,2].hist([data_enhanced[~hp_mask]['price'], data_enhanced[hp_mask]['price']], 
               bins=30, alpha=0.7, label=['Regular', 'High Performance'], color=['lightgreen', 'red'])
axes[1,2].set_title('Price Distribution: Performance Level', fontweight='bold')
axes[1,2].set_xlabel('Price ($)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].legend()

plt.tight_layout()
plt.show()

# %%
# Prepare data for modeling
print("🔧 Preparing data for modeling...")

# Clean data
data_clean = data_enhanced.dropna(subset=['price'])
data_clean = data_clean[data_clean['price'] > 0]  # Remove zero prices

# Remove extreme outliers
price_low = data_clean['price'].quantile(0.01)
price_high = data_clean['price'].quantile(0.99)
data_clean = data_clean[(data_clean['price'] >= price_low) & (data_clean['price'] <= price_high)]

print(f"Clean dataset: {data_clean.shape[0]} rows")

# Split features and target
X_raw = data_clean.drop('price', axis=1).select_dtypes(include=[np.number, 'object'])
X_enhanced = data_clean.drop('price', axis=1)
y = data_clean['price']

# Train-test split
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
X_enh_train, X_enh_test, _, _ = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

print(f"Training set: {X_raw_train.shape[0]} samples")
print(f"Test set: {X_raw_test.shape[0]} samples")

# %%
# Create preprocessing pipelines
def create_pipeline(model, enhanced=False):
    """Create preprocessing pipeline."""
    if enhanced:
        # Enhanced features
        numeric_features = ['0', '1', '2', '3', '4', 'engine_capacity_cc', 'horsepower',
                          'brand_tier', 'power_category', 'engine_category', 'power_per_liter',
                          'brand_power_interaction', 'anon_sum', 'anon_mean', 'anon_std',
                          'is_luxury', 'is_high_performance']
        categorical_features = ['brand', 'model', 'trim', 'body_type', 'fuel_type', 
                              'transmission_type', 'exterior_color', 'interior_color', 
                              'warranty', 'city', 'seller_type']
    else:
        # Raw features only
        numeric_features = ['0', '1', '2', '3', '4']
        categorical_features = ['brand', 'model', 'trim', 'body_type', 'fuel_type', 
                              'transmission_type', 'engine_capacity_cc', 'horsepower',
                              'exterior_color', 'interior_color', 'warranty', 'city', 'seller_type']
    
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

# %%
# Define models to test - Only good performing models (>50% R²)
models = [
    ('Extra Trees Optimized', ExtraTreesRegressor(
        n_estimators=500,           # More trees for better performance
        max_depth=None,             # No depth limit
        min_samples_split=2,        # Allow more splitting
        min_samples_leaf=1,         # More granular predictions
        max_features='sqrt',        # Use subset of features
        bootstrap=False,            # Use all samples
        random_state=42, 
        n_jobs=-1
    )),
    ('Random Forest Pro', RandomForestRegressor(
        n_estimators=300,           # More trees
        max_depth=None,             # No depth limit
        min_samples_split=3,        # Slightly more conservative
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42, 
        n_jobs=-1
    )),
    ('Gradient Boosting Pro', GradientBoostingRegressor(
        n_estimators=300,           # More estimators
        max_depth=6,                # Deeper trees
        learning_rate=0.1,          # Good learning rate
        subsample=0.8,              # Use 80% of samples
        max_features='sqrt',
        random_state=42
    )),
    ('Ridge Regression', Ridge(alpha=10.0)),  # Baseline linear model
    ('Lasso Regression', Lasso(alpha=1.0)),   # Another baseline
    ('Extra Trees Ensemble', ExtraTreesRegressor(
        n_estimators=800,           # Even more trees
        max_depth=25,               # Limit depth slightly
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,          # Use all features
        bootstrap=False,
        random_state=42, 
        n_jobs=-1
    ))
]

print(f"🤖 Testing {len(models)} high-quality models...")

# %%
# Train and evaluate models
results = []

print("\n📊 Model Performance Comparison")
print("=" * 80)
print(f"{'Model':<20} {'Raw Data R²':<15} {'Enhanced R²':<15} {'Improvement':<15}")
print("-" * 80)

for name, model in models:
    # Raw data performance
    pipeline_raw = create_pipeline(model, enhanced=False)
    pipeline_raw.fit(X_raw_train, y_train)
    y_pred_raw = pipeline_raw.predict(X_raw_test)
    r2_raw = r2_score(y_test, y_pred_raw)
    mae_raw = mean_absolute_error(y_test, y_pred_raw)
    
    # Enhanced data performance
    pipeline_enh = create_pipeline(model, enhanced=True)
    pipeline_enh.fit(X_enh_train, y_train)
    y_pred_enh = pipeline_enh.predict(X_enh_test)
    r2_enh = r2_score(y_test, y_pred_enh)
    mae_enh = mean_absolute_error(y_test, y_pred_enh)
    
    improvement = r2_enh - r2_raw
    
    results.append({
        'model': name,
        'r2_raw': r2_raw,
        'mae_raw': mae_raw,
        'r2_enhanced': r2_enh,
        'mae_enhanced': mae_enh,
        'improvement': improvement
    })
    
    print(f"{name:<20} {r2_raw*100:>6.1f}%        {r2_enh*100:>6.1f}%        {improvement*100:>+6.1f}%")

# %%
# Results Analysis and Visualization
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2_enhanced', ascending=False)

print(f"\n🏆 TOP 3 BEST MODELS (with Feature Engineering)")
print("=" * 60)
for i, row in results_df.head(3).iterrows():
    print(f"{i+1}. {row['model']}")
    print(f"   Enhanced R²: {row['r2_enhanced']*100:.1f}%")
    print(f"   Enhanced MAE: ${row['mae_enhanced']:,.0f}")
    print(f"   Improvement: {row['improvement']*100:+.1f}%")
    print()

# %%
# Comparison with skops models (our simple models)
skops_results = [
    {'model': 'Extra Trees (Skops)', 'r2': 0.6478, 'mae': 51354},
    {'model': 'Random Forest (Skops)', 'r2': 0.6189, 'mae': 57887}, 
    {'model': 'Gradient Boosting (Skops)', 'r2': 0.5271, 'mae': 73980}
]

print(f"📁 SKOPS MODELS PERFORMANCE (Raw Data Only)")
print("=" * 50)
for i, model in enumerate(skops_results, 1):
    print(f"{i}. {model['model']}")
    print(f"   R²: {model['r2']*100:.1f}%")
    print(f"   MAE: ${model['mae']:,}")
    print()

# %%
# Create comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Analysis: Raw vs Enhanced Features', fontsize=16, fontweight='bold')

# R² comparison
model_names = [r['model'] for r in results]
r2_raw_vals = [r['r2_raw'] for r in results]
r2_enh_vals = [r['r2_enhanced'] for r in results]

x = np.arange(len(model_names))
width = 0.35

axes[0,0].bar(x - width/2, [r*100 for r in r2_raw_vals], width, label='Raw Data', alpha=0.8, color='lightcoral')
axes[0,0].bar(x + width/2, [r*100 for r in r2_enh_vals], width, label='Enhanced Features', alpha=0.8, color='skyblue')
axes[0,0].set_title('R² Score Comparison', fontweight='bold')
axes[0,0].set_xlabel('Models')
axes[0,0].set_ylabel('R² Score (%)')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Improvement visualization
improvements = [r['improvement']*100 for r in results]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
axes[0,1].bar(model_names, improvements, color=colors, alpha=0.7)
axes[0,1].set_title('Performance Improvement with Feature Engineering', fontweight='bold')
axes[0,1].set_xlabel('Models')
axes[0,1].set_ylabel('R² Improvement (%)')
axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
axes[0,1].grid(True, alpha=0.3)

# Top models comparison
top_3_enhanced = results_df.head(3)
skops_names = [s['model'] for s in skops_results]
skops_r2 = [s['r2']*100 for s in skops_results]

axes[1,0].bar(range(len(top_3_enhanced)), top_3_enhanced['r2_enhanced']*100, 
              alpha=0.8, color='gold', label='Best Enhanced Models')
axes[1,0].bar(range(len(skops_results)), skops_r2, 
              alpha=0.8, color='lightgray', label='Skops Models (Raw)')
axes[1,0].set_title('Top 3 Enhanced vs Skops Models', fontweight='bold')
axes[1,0].set_xlabel('Model Rank')
axes[1,0].set_ylabel('R² Score (%)')
axes[1,0].set_xticks(range(max(len(top_3_enhanced), len(skops_results))))
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Feature engineering impact summary
categories = ['Raw Data\nAverage', 'Enhanced\nAverage', 'Best Enhanced\nModel']
avg_raw = np.mean([r['r2_raw'] for r in results]) * 100
avg_enh = np.mean([r['r2_enhanced'] for r in results]) * 100
best_enh = max([r['r2_enhanced'] for r in results]) * 100

values = [avg_raw, avg_enh, best_enh]
colors_summary = ['lightcoral', 'skyblue', 'gold']

bars = axes[1,1].bar(categories, values, color=colors_summary, alpha=0.8)
axes[1,1].set_title('Feature Engineering Impact Summary', fontweight='bold')
axes[1,1].set_ylabel('R² Score (%)')
axes[1,1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Performance Summary Table
print(f"\n📋 COMPREHENSIVE PERFORMANCE SUMMARY")
print("=" * 100)
print(f"{'Rank':<5} {'Model':<20} {'Raw R²':<10} {'Enhanced R²':<12} {'Raw MAE':<12} {'Enhanced MAE':<15} {'Improvement'}")
print("-" * 100)

for i, row in results_df.iterrows():
    print(f"{i+1:<5} {row['model']:<20} {row['r2_raw']*100:>6.1f}%    "
          f"{row['r2_enhanced']*100:>6.1f}%      ${row['mae_raw']:>8,.0f}    "
          f"${row['mae_enhanced']:>10,.0f}     {row['improvement']*100:>+6.1f}%")

# %%
# Key Insights and Conclusions
print(f"\n🎯 KEY INSIGHTS")
print("=" * 50)

best_model = results_df.iloc[0]
avg_improvement = np.mean([r['improvement'] for r in results]) * 100
models_above_70 = len([r for r in results if r['r2_enhanced'] > 0.70])
models_above_80 = len([r for r in results if r['r2_enhanced'] > 0.80])

print(f"✅ Best performing model: {best_model['model']} ({best_model['r2_enhanced']*100:.1f}% R²)")
print(f"✅ Average improvement from feature engineering: {avg_improvement:+.1f}%")
print(f"✅ Models achieving >70% R²: {models_above_70}/{len(results)}")
print(f"✅ Models achieving >80% R²: {models_above_80}/{len(results)}")

if best_model['r2_enhanced'] >= 0.70:
    print(f"🎉 SUCCESS: Achieved 70%+ R² target!")
if best_model['r2_enhanced'] >= 0.80:
    print(f"🚀 EXCELLENT: Achieved 80%+ R² target!")

print(f"\n💡 Feature engineering improved model performance by an average of {avg_improvement:.1f}%")
print(f"💡 The best model is {((best_model['r2_enhanced'] - 0.6478) * 100):+.1f}% better than our best skops model")

print(f"\n✨ Analysis Complete! ✨") 