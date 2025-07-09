# 🚗 Car Price Prediction Project

A comprehensive machine learning project that compares different models with and without feature engineering to predict car prices.

## 📁 Project Files

### Core Files:
- `topic21_v40_train.csv` - Training dataset with car information
- `create_three_models.py` - Creates 3 simple models (No feature engineering)
- `test_models.py` - Tests the 3 skops model files
- `complete_car_price_analysis.py` - Full analysis with feature engineering + visualizations

### Generated Files:
- `model_1.skops` - Extra Trees model (Raw data)
- `model_2.skops` - Random Forest model (Raw data) 
- `model_3.skops` - Gradient Boosting model (Raw data)

## 🚀 Quick Start

### Step 1: Install Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn skops
```

### Step 2: Get the Dataset
Make sure you have `topic21_v40_train.csv` in the same folder as the Python files.

### Step 3: Run the Analysis

#### Option A: Simple Models (Raw Data Only)
```bash
# Create 3 simple models
python create_three_models.py

# Test the models
python test_models.py
```

#### Option B: Complete Analysis (Jupyter/Colab)
1. Copy the content from `complete_car_price_analysis.py`
2. Paste it into Jupyter Notebook or Google Colab
3. Run each cell sequentially

## 📊 What Each File Does

### `create_three_models.py`
- Creates 3 different ML models using raw data only
- No feature engineering - just basic preprocessing
- Saves models as .skops files
- **Expected Performance**: 50-65% R²

### `test_models.py`  
- Loads and tests the 3 .skops model files
- Shows performance metrics for each model
- Ranks models by R² score

### `complete_car_price_analysis.py`
- **Complete analysis with feature engineering**
- Tests 6 high-quality ML algorithms
- Compares raw data vs enhanced features
- Creates beautiful visualizations
- **Expected Performance**: 70-85% R²

## 🎯 What You'll See

### 1. Data Visualizations
- Price distribution charts
- Brand vs Price analysis
- Body type, fuel type comparisons
- Horsepower correlations
- Enhanced feature analysis

### 2. Model Comparisons
- Raw data performance vs Enhanced features
- Before/after feature engineering improvements
- Top 3 best models
- Comparison with simple skops models

### 3. Performance Metrics
- R² scores (target: 70%+)
- Mean Absolute Error (MAE)
- Improvement percentages

## 📈 Expected Results

| Model Type | Raw Data R² | Enhanced R² | Improvement |
|------------|-------------|-------------|-------------|
| Simple Models | 50-65% | N/A | N/A |
| Enhanced Models | 50-65% | 70-85% | +15-25% |

## 🔧 For Jupyter Notebook Users

1. **Copy-Paste Friendly**: The `complete_car_price_analysis.py` file is formatted with `# %%` cell separators
2. **Just copy the entire content and paste into Jupyter**
3. **Run cells one by one** to see step-by-step analysis
4. **All visualizations will appear inline**

## 🎨 Key Features

### Visualizations Include:
- **Raw Data Analysis**: Price distributions, brand comparisons
- **Enhanced Features**: Brand tiers, power categories, correlation heatmaps
- **Model Performance**: Before/after comparisons, improvement charts
- **Summary Charts**: Best models, feature engineering impact

### Machine Learning Models:
1. **Extra Trees Optimized** (500+ estimators)
2. **Random Forest Pro** (300+ estimators)
3. **Gradient Boosting Pro** (optimized parameters)
4. **Ridge Regression** (baseline linear model)
5. **Lasso Regression** (L1 regularization baseline)
6. **Extra Trees Ensemble** (800+ estimators for maximum performance)

## 💡 Key Insights You'll Discover

- How feature engineering improves model performance
- Which car features most influence price
- Comparison between luxury and mainstream brands
- Impact of horsepower and engine size on pricing
- Best performing ML algorithms for this dataset

## 🆘 Troubleshooting

### Common Issues:

1. **Package Missing**: 
   ```bash
   pip install [package-name]
   ```

2. **File Not Found**: Make sure `topic21_v40_train.csv` is in the same folder

3. **Jupyter Issues**: Make sure to run cells in order from top to bottom

4. **Memory Issues**: If running on low-memory systems, reduce `n_estimators` in models

## 🎓 Learning Objectives

After running this project, you'll understand:
- Data preprocessing and feature engineering
- Multiple ML algorithm comparison
- Model evaluation metrics (R², MAE)
- Data visualization with matplotlib/seaborn
- The impact of feature engineering on model performance

## 📝 Notes

- The dataset contains car information with price as the target variable
- Feature engineering creates new features like brand tiers, power categories, etc.
- All models use the same train/test split for fair comparison
- Visualizations are designed to be publication-ready

## 🎉 Have Fun!

This project demonstrates the power of feature engineering and proper model comparison. Enjoy exploring the data and seeing how different approaches affect model performance!

---

**Created with ❤️ for learning machine learning**
