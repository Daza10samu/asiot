import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")

mlflow.set_experiment("iris-classification-lab")


def train_model(solver="lbfgs", max_iter=1000, C=1.0):
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"lr_{solver}_C{C}"):
        params = {
            "solver": solver,
            "max_iter": max_iter,
            "C": C,
            "random_state": 42
        }
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri,
            "iris-classifier"
        )

        return mlflow.active_run().info.run_id, registered_model.version


def test_model(model_name, model_version):
    """Загрузить и протестировать зарегистрированную модель"""

    model_uri = f"models:/{model_name}/{model_version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    test_samples = [
        X_test[0:5],
        X_test[5:10],
        X_test[10:15]
    ]

    for i, sample in enumerate(test_samples, 1):
        predictions = loaded_model.predict(sample)
        actual = y_test[(i - 1) * 5:i * 5]
        test_accuracy = accuracy_score(actual, predictions)

        print(f"\nТест {i}:")
        print(f"  Предсказания: {predictions}")
        print(f"  Фактические: {actual}")
        print(f"  Точность: {test_accuracy:.4f}")


if __name__ == "__main__":
    print("=== Обучение нескольких моделей ===\n")

    configs = [
        {"solver": "lbfgs", "max_iter": 1000, "C": 1.0},
        {"solver": "liblinear", "max_iter": 1000, "C": 0.5},
        {"solver": "saga", "max_iter": 2000, "C": 2.0}
    ]

    run_ids = []
    for config in configs:
        print(f"\nОбучение с конфигурацией: {config}")
        run_id, version = train_model(**config)
        run_ids.append((run_id, version))

    print("\n\n=== Тестирование последней модели ===")
    latest_version = run_ids[-1][1]
    test_model("iris-classifier", latest_version)
