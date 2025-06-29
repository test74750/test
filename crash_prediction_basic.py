import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your data
data = pd.read_csv('crash_data.csv')  # Assume your data is in this CSV

# Visualize the data
plt.plot(data['round'], data['crash_value'])
plt.xlabel('Round')
plt.ylabel('Crash Value')
plt.title('Crash Value per Round')
plt.show()

# Feature engineering (example: use previous N rounds to predict next)
N = 5  # Number of previous rounds to use as features
features = []
targets = []

for i in range(N, len(data)):
    features.append(data['crash_value'].iloc[i-N:i].values)
    targets.append(data['crash_value'].iloc[i])

X = np.array(features)
y = np.array(targets)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

# Calculate accuracy (e.g., mean squared error)
from sklearn.metrics import mean_squared_error
print('MSE:', mean_squared_error(y_test, predictions))