from flask import Flask, render_template, request, jsonify
import requests
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("models/crypto_model.pkl")

# CoinGecko API endpoint
API_URL = "https://api.coingecko.com/api/v3/coins/markets"
SUPPORTED_COINS = {
    "bitcoin": "Bitcoin (BTC)",
    "ethereum": "Ethereum (ETH)",
    "ripple": "Ripple (XRP)",
    "cardano": "Cardano (ADA)",
    "solana": "Solana (SOL)"
}

def fetch_crypto_data(coin_id):
    params = {
        "vs_currency": "usd",
        "ids": coin_id
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return {
            "open": data.get("current_price"),
            "high": data.get("high_24h"),
            "low": data.get("low_24h"),
            "volume": data.get("total_volume"),
            "market_cap": data.get("market_cap")
        }
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    crypto_data = {}
    selected_coin = "bitcoin"

    if request.method == "POST":
        selected_coin = request.form["crypto"]
        crypto_data = fetch_crypto_data(selected_coin)
        if crypto_data:
            features = np.array([[crypto_data["open"], crypto_data["high"], crypto_data["low"], crypto_data["volume"], crypto_data["market_cap"]]])
            prediction = model.predict(features)[0]

    return render_template("index.html", coins=SUPPORTED_COINS, selected_coin=selected_coin, data=crypto_data, prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT Render assigns
    app.run(host="0.0.0.0", port=port)        # Bind to 0.0.0.0 so Render can detect it
