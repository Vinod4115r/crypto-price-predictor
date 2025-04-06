import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Sample mock training data
data = {
    'open': [20000, 21000, 19000, 22000, 21500],
    'high': [20500, 21500, 19500, 22500, 21800],
    'low': [19800, 20800, 18800, 21800, 21300],
    'volume': [300000000, 310000000, 290000000, 330000000, 320000000],
    'market_cap': [400000000000, 410000000000, 390000000000, 420000000000, 415000000000],
    'close': [20200, 21200, 19200, 22200, 21600]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['open', 'high', 'low', 'volume', 'market_cap']]
y = df['close']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/crypto_model.pkl")

print("Model trained and saved to models/crypto_model.pkl")
