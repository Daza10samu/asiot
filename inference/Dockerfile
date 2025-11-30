FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir \
    mlflow==2.15.0 \
    flask==3.0.0 \
    pandas==2.2.0 \
    scikit-learn==1.3.1 \
    gunicorn==21.2.0

# Копирование приложения
COPY inference_service.py .

# Переменные окружения
ENV MLFLOW_TRACKING_URI="http://192.168.2.9:5000"
ENV MODEL_URI="models:/wine-classifier-rf/Staging"
ENV MLFLOW_TRACKING_INSECURE_TLS="true"

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()"

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "inference_service:app"]
