# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate Synthetic Dataset
np.random.seed(42)  # for reproducibility

num_samples = 1000
data = pd.DataFrame({
    'industry_type': np.random.choice(['chemical', 'steel', 'textile', 'electronics'], num_samples),
    'production_level': np.random.uniform(100, 10000, num_samples),  # units per month
    'energy_consumption': np.random.uniform(5000, 50000, num_samples),  # kWh per month
    'fuel_type': np.random.choice(['coal', 'natural_gas', 'renewable'], num_samples),
    'operational_hours': np.random.uniform(100, 720, num_samples)  # hours per month
})

# Assume emissions depend on these factors (synthetic relationship)
emission_factors = {
    'chemical': 2.5,
    'steel': 3.0,
    'textile': 1.5,
    'electronics': 1.0,
    'coal': 3.5,
    'natural_gas': 2.0,
    'renewable': 0.5
}

data['emissions'] = (
    data['production_level'] * data['industry_type'].map(emission_factors) * 0.001 +
    data['energy_consumption'] * data['fuel_type'].map(emission_factors) * 0.0001 +
    data['operational_hours'] * 0.1 +
    np.random.normal(0, 10, num_samples)  # random noise
)

# 2. Preprocessing Data
# Convert categorical variables to dummy variables
data_processed = pd.get_dummies(data, columns=['industry_type', 'fuel_type'], drop_first=True)

X = data_processed.drop('emissions', axis=1)
y = data_processed['emissions']

# 3. Split dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Predictive Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')

# 6. Making Predictions for New Data
# Ensuring all features match training set exactly
new_industry_data = pd.DataFrame({
    'production_level': [5000],
    'energy_consumption': [25000],
    'operational_hours': [400],
    'industry_type_electronics': [1],
    'industry_type_steel': [0],
    'industry_type_textile': [0],
    'fuel_type_natural_gas': [0],


  
    'fuel_type_renewable': [1]
})

predicted_emission = model.predict(new_industry_data)
print(f'Predicted Emission: {predicted_emission[0]:.2f} units')
