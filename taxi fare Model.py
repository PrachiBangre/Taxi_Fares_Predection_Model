import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Set the number of samples
num_samples = 3000

# Generate random pickup_datetime
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-12-31')
pickup_datetime = pd.date_range(start=start_date, end=end_date, periods=num_samples)

# Generate random pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude
pickup_longitude = np.random.uniform(low=-74.3, high=-73.7, size=num_samples)
pickup_latitude = np.random.uniform(low=40.5, high=41.0, size=num_samples)
dropoff_longitude = np.random.uniform(low=-74.3, high=-73.7, size=num_samples)
dropoff_latitude = np.random.uniform(low=40.5, high=41.0, size=num_samples)

# Generate random pickup_hour
pickup_hour = np.random.randint(0, 24, size=num_samples)

# Generate random pickup_day (0: Monday, 1: Tuesday, ..., 6: Sunday)
pickup_day = np.random.randint(0, 7, size=num_samples)

# Generate random fare_amount
fare_amount = np.random.uniform(low=5, high=50, size=num_samples)

# Create DataFrame
data = pd.DataFrame({
    'pickup_longitude': pickup_longitude,
    'pickup_latitude': pickup_latitude,
    'dropoff_longitude': dropoff_longitude,
    'dropoff_latitude': dropoff_latitude,
    'pickup_datetime': pickup_datetime,
    'pickup_hour': pickup_hour,
    'pickup_day': pickup_day,
    'fare_amount': fare_amount
})

# Load the dataset
taxi_data = data

# Feature engineering: Extract relevant features from pickup_datetime
taxi_data['pickup_hour'] = taxi_data['pickup_datetime'].dt.hour
taxi_data['pickup_day'] = taxi_data['pickup_datetime'].dt.dayofweek

# Selecting features and target variable
X = taxi_data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day']]
y = taxi_data['fare_amount']

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Feature importance
feature_importance = rf_model.feature_importances_
feature_names = X.columns

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Maximizing Taxi Fare')
plt.show()
