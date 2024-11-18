# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Load Dataset
file_path = r'C:\\Users\\ketan\\OneDrive\\Desktop\\archive\\data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Convert 'Date_Time' to datetime and extract Day, Month, Year
df['Date_Time'] = pd.to_datetime(df['Date_Time'], dayfirst=True, errors='coerce')
df['Day'] = df['Date_Time'].dt.day
df['Month'] = df['Date_Time'].dt.month
df['Year'] = df['Date_Time'].dt.year

# Drop unnecessary columns
df = df.drop(['id', 'Date_Time'], axis=1)

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Impute Missing Values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Preview the cleaned dataset
print("\nCleaned Dataset:")
print(df_imputed.head())

# Data Visualization
## Box Plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_imputed[['Average_Hourly', 'Total_Count', 'VehicleCount']])
plt.title("Box Plot of Numerical Features")
plt.show()

## Histogram
df_imputed[['Average_Hourly', 'Total_Count', 'VehicleCount']].hist(bins=15, figsize=(12, 6))
plt.suptitle("Histogram of Numerical Features")
plt.show()

## Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df_imputed['Total_Count'], df_imputed['VehicleCount'], alpha=0.7)
plt.title("Scatter Plot: Total Count vs Vehicle Count")
plt.xlabel("Total Count")
plt.ylabel("Vehicle Count")
plt.show()

# Pearson Correlation
correlation_matrix = df_imputed.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Pearson Correlation Matrix")
plt.show()

# Print Correlation between Total Count and Vehicle Count
correlation_value = correlation_matrix['VehicleCount']['Total_Count']
print(f"Correlation between Total_Count and VehicleCount: {correlation_value}")

# Feature Identification
X = df_imputed[['Day', 'Month', 'Year', 'Average_Hourly', 'Total_Count']]
y = df_imputed['VehicleCount']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Save the Model
joblib.dump(model, 'vehicle_count_prediction_model.pkl')

# Visualize Actual vs Predicted Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
plt.title("Actual vs Predicted Vehicle Counts")
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.show()
