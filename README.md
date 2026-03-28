# California Housing Price Prediction

A machine learning project that predicts median house values in California using the classic California Housing dataset.

## Overview

This notebook explores the California Housing dataset, applies feature engineering and preprocessing, then compares a **Linear Regression** baseline against a **Random Forest** model to predict house prices.

## Dataset

- **Source:** `housing.csv` (California Housing dataset)
- **Size:** 20,640 entries × 10 features
- **Target:** `median_house_value` (in USD)

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Block population |
| `households` | Number of households |
| `median_income` | Median income (in tens of thousands USD) |
| `ocean_proximity` | Categorical: distance to ocean |



## Methodology

### 1. Data Cleaning
- Dropped 207 rows with missing `total_bedrooms` values

### 2. Exploratory Data Analysis
- Histograms of all features
- Correlation heatmaps
- Geographic scatter plot (latitude vs longitude colored by price) — confirms coastal houses are significantly more expensive

### 3. Feature Engineering
- **Log transform** applied to `total_rooms`, `total_bedrooms`, `population`, `households` to reduce right skew
- **One-hot encoding** of `ocean_proximity` (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`)
- **New features:**
  - `bedroom_ratio` = `total_bedrooms / total_rooms`
  - `household_rooms` = `total_rooms / households`

### 4. Preprocessing
- Standard scaling (`StandardScaler`) fitted on training data only, applied to both train and test sets

## Models & Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.6687 | — |
| Random Forest (default) | 0.8218 | — |
| **Random Forest (tuned, 100 estimators)** | **0.8194** | **$49,690** |

Hyperparameter tuning was done via `GridSearchCV` with 5-fold cross-validation over `n_estimators` ∈ {3, 10, 30} and `max_features` ∈ {2, 4, 6, 8}. Best params found: `max_features=8, n_estimators=30`.

The final model uses `n_estimators=100` for better stability.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```python
# Predict price for a new house
new_house = x_test.iloc[0]
new_house_scaled = scaler.transform([new_house])
predicted_price = forest.predict(new_house_scaled)[0]
print(f"Predicted price: ${predicted_price:.2f}")
```

## Key Findings

- **Median income** is the strongest predictor of house value
- **Location** (latitude/longitude + ocean proximity) plays a major role — coastal blocks command significantly higher prices
- Log-transforming skewed features and adding ratio features meaningfully improved model performance
- Random Forest outperforms Linear Regression by ~15 percentage points in R²
