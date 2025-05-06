
import json
import websocket
import numpy as np
import pickle
import time
import requests
from sklearn.linear_model import LogisticRegression

API_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
API_TOKEN = "cJ5Llq9gszcefBI"
TELEGRAM_BOT_TOKEN = "8072219007:AAFAQku9cqXGyY8nomx1wnyPQqY0X_nzkgU"
TELEGRAM_CHAT_ID = "7530463006"
TRADE_AMOUNT = 100  
COOLDOWN_TIME = 120  
RECONNECT_DELAY = 10
last_trade_time = 0  
price_history = []  

model = LogisticRegression()

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("[INFO] Loaded pre-trained model.")
except FileNotFoundError:
    print("[INFO] No pre-trained model found. Training required.")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("[ERROR] Failed to send Telegram message:", str(e))

def place_trade(ws, prediction, confidence):
    global last_trade_time, COOLDOWN_TIME
    current_time = time.time()
    time_since_last_trade = current_time - last_trade_time

    if time_since_last_trade < COOLDOWN_TIME:
        remaining_time = int(COOLDOWN_TIME - time_since_last_trade)
        print(f"[INFO] Cooldown active. {remaining_time} seconds remaining. Skipping trade.")
        return

    if confidence < 0.6:
        print("[INFO] Confidence too low. Skipping trade.")
        return

    trade_request = json.dumps({
        "buy": 1,
        "price": TRADE_AMOUNT,
        "parameters": {
            "symbol": "R_100",
            "contract_type": "CALL" if prediction == 1 else "PUT",
            "amount": TRADE_AMOUNT,
            "basis": "stake",
            "currency": "USD",
            "duration": 1,  # Added duration parameter to fix ContractCreationFailure error
            "duration_unit": "t"  # 1 tick duration
        }
    })
    print("[INFO] Placing trade:", trade_request)
    ws.send(trade_request)
    send_telegram_message(f"Trade placed: {trade_request}")
    last_trade_time = time.time()
    COOLDOWN_TIME = 120 if prediction == 1 else 150

def on_message(ws, message):
    print("[MESSAGE] Received:", message)
    try:
        data = json.loads(message)
        if "tick" in data:
            features = extract_features(data)
            if features is not None:
                price_history.append(features)
                if len(price_history) > 100:
                    price_history.pop(0)
                train_model()
                
                if len(price_history) >= 10:
                    features = np.array(features).reshape(1, -1)
                    prediction = model.predict(features)[0]
                    confidence = model.predict_proba(features)[0][prediction]
                    print(f"[PREDICTION] {prediction} with confidence {confidence:.2f}")
                    send_telegram_message(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
                    place_trade(ws, prediction, confidence)
                else:
                    print("[INFO] Not enough data to make a prediction yet.")
        elif "authorize" in data and data["authorize"]:
            tick_subscribe = json.dumps({"ticks": "R_100"})
            ws.send(tick_subscribe)
            print("[INFO] Subscribed to tick data for R_100")
        elif "buy" in data:
            if data["buy"].get("buy_price", None) is not None:
                global last_trade_time
                last_trade_time = time.time()
                print("[INFO] Trade executed successfully. Cooldown reset.")
    except Exception as e:
        print("[ERROR] Prediction failed:", str(e))

def on_error(ws, error):
    print("[ERROR]", error)
    send_telegram_message(f"[ERROR] {error}")
    time.sleep(RECONNECT_DELAY)

def on_close(ws, close_status_code, close_msg):
    print("[CLOSED] Status:", close_status_code, "Message:", close_msg)
    send_telegram_message("[INFO] WebSocket closed. Reconnecting...")
    time.sleep(RECONNECT_DELAY)
    start_websocket()

def on_open(ws):
    print("[CONNECTED] WebSocket is open")
    try:
        auth_request = json.dumps({"authorize": API_TOKEN})
        ws.send(auth_request)
        print("[INFO] Sent authorization request...")
    except Exception as e:
        print("[ERROR] Failed to send auth request:", str(e))

def extract_features(data):
    tick = data.get("tick", {})
    current_price = tick.get("quote")
    if current_price is None:
        print("[ERROR] Missing quote data in tick.")
        return None
    features = [
        current_price,
        current_price / 100,
        current_price * 10,
        current_price - 100,
        current_price + 50,
        np.sin(current_price),
        np.cos(current_price)
    ]
    return features

def train_model():
    global model
    if len(price_history) >= 10:
        X_train = np.array(price_history[:-1])
        y_train = np.random.randint(0, 2, len(X_train))  
        model.fit(X_train, y_train)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("[INFO] Model retrained with real-time data.")

def start_websocket():
    print("[INFO] Starting WebSocket...")
    while True:
        try:
            ws = websocket.WebSocketApp(API_URL,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
            ws.on_open = on_open
            ws.run_forever()
        except Exception as e:
            print("[ERROR] WebSocket connection failed:", str(e))
            send_telegram_message(f"[ERROR] WebSocket connection failed: {e}")
            time.sleep(RECONNECT_DELAY)

start_websocket()
