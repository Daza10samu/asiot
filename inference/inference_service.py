from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import os
import pandas as pd
import time

app = Flask(__name__)

# Счетчики для метрик
prediction_counter = 0
error_counter = 0
start_time = time.time()

# Загрузка модели при старте
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.labs.itmo.loc")
MODEL_URI = os.getenv("MODEL_URI", "models:/wine-classifier-rf/Staging")

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"Loading model from {MODEL_URI}...")
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route('/invocations', methods=['POST'])
def predict():
    global prediction_counter, error_counter

    if model is None:
        error_counter += 1
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        # Поддержка разных форматов
        if 'dataframe_split' in data:
            df = pd.DataFrame(
                data['dataframe_split']['data'],
                columns=data['dataframe_split']['columns']
            )
        elif 'instances' in data:
            df = pd.DataFrame(data['instances'])
        else:
            df = pd.DataFrame(data)

        # Предсказание
        predictions = model.predict(df)
        prediction_counter += len(predictions)

        return jsonify({
            "predictions": predictions.tolist()
        }), 200

    except Exception as e:
        error_counter += 1
        return jsonify({
            "error": str(e)
        }), 400


@app.route('/metrics', methods=['GET'])
def metrics():
    uptime = time.time() - start_time

    metrics_text = f"""# HELP wine_model_predictions_total Total predictions made
# TYPE wine_model_predictions_total counter
wine_model_predictions_total {prediction_counter}

# HELP wine_model_errors_total Total prediction errors
# TYPE wine_model_errors_total counter
wine_model_errors_total {error_counter}

# HELP wine_model_uptime_seconds Service uptime in seconds
# TYPE wine_model_uptime_seconds gauge
wine_model_uptime_seconds {uptime:.2f}

# HELP wine_model_info Model information
# TYPE wine_model_info gauge
wine_model_info{{model_name="wine-classifier-rf",version="1"}} 1
"""
    return metrics_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
