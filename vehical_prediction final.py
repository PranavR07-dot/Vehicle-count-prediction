# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset from the specific path (replace with your actual path if necessary)
df = pd.read_csv(r'C:\\Users\\ketan\\OneDrive\\Desktop\\archive\\data.csv')

# Print the column names to confirm
print("Columns in the dataset:", df.columns)

# Step 1: Preprocess Data
# Convert 'Date_Time' to datetime format with dayfirst=True to handle mixed date/time formats
df['Date_Time'] = pd.to_datetime(df['Date_Time'], dayfirst=True, errors='coerce')

# Check for any errors in parsing
print("\nParsed 'Date_Time' column:")
print(df['Date_Time'].head())

# Extract useful features: Day, Month, and Year
df['Day'] = df['Date_Time'].dt.day
df['Month'] = df['Date_Time'].dt.month
df['Year'] = df['Date_Time'].dt.year

# Drop the 'id' and 'Date_Time' columns
df = df.drop(['id', 'Date_Time'], axis=1)

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Step 2: Impute Missing Values
# Create an imputer object with strategy 'mean' to replace NaN values with the mean of the column
imputer = SimpleImputer(strategy='mean')

# Impute missing values for the feature columns (X)
X = df[['Day', 'Month', 'Year', 'Average_Hourly', 'Total_Count']]
X_imputed = imputer.fit_transform(X)

# Impute missing values for the target column (y)
y = df['VehicleCount']
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1))  # Reshape for single column target

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Visualize the Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
plt.xlabel('Actual Vehicle Count')
plt.ylabel('Predicted Vehicle Count')
plt.title('Actual vs Predicted Vehicle Counts')
plt.show()

# Step 8: Save the Model (Optional)
joblib.dump