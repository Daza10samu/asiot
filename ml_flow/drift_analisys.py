import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

from evidently import Report
from evidently.metrics import DriftedColumnsCount, DatasetMissingValueCount, ValueDrift

# Подключение к MLflow
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")


def load_iris_data():
    """Загрузка и подготовка данных Iris"""
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    data = pd.concat([X, pd.DataFrame(y, columns=['target'])], axis=1)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    return data


def simulate_drift_data(reference_data, drift_type='shift'):
    """Создание данных с дрифтом"""
    drifted_data = reference_data.copy()

    if drift_type == 'shift':
        drifted_data['sepal_length'] = drifted_data['sepal_length'] + 1.5
        drifted_data['sepal_width'] = drifted_data['sepal_width'] * 0.8
    elif drift_type == 'missing':
        mask = np.random.random(len(drifted_data)) < 0.15
        drifted_data.loc[mask, 'petal_length'] = np.nan
    elif drift_type == 'target':
        drifted_data['target'] = drifted_data['target'].replace({0: 2, 2: 0})

    return drifted_data


def generate_drift_report(reference_data, current_data, report_name="data_drift_report.html"):
    """Генерация HTML отчета о дрифте с новым API Evidently 0.7+"""

    report = Report(metrics=[
        DriftedColumnsCount(),
        DatasetMissingValueCount(),
        ValueDrift(column='target'),
        ValueDrift(column='sepal_length'),
        ValueDrift(column='sepal_width'),
        ValueDrift(column='petal_length'),
        ValueDrift(column='petal_width')
    ])

    result = report.run(current_data=current_data, reference_data=reference_data)

    result.save_html(report_name)
    print(f"✅ Отчет сохранен: {report_name}")

    # Возвращаем словарь результатов
    return result.dict()


def run_drift_tests(reference_data, current_data):
    """Программное тестирование на дрифт"""

    report = Report(metrics=[
        DriftedColumnsCount(),
        DatasetMissingValueCount()
    ])

    result = report.run(current_data=current_data, reference_data=reference_data)
    results_dict = result.dict()

    print("\n=== Результаты тестирования ===")

    has_drift = False
    n_drifted = 0

    for metric in results_dict.get('metrics', []):
        if metric.get('metric') == 'DriftedColumnsCount':
            metric_result = metric.get('result', {})
            has_drift = metric_result.get('dataset_drift', False)
            n_drifted = metric_result.get('number_of_drifted_columns', 0)
            drift_share = metric_result.get('drift_share', 0)

            print(f"Dataset Drift обнаружен: {has_drift}")
            print(f"Количество признаков с дрифтом: {n_drifted}")
            print(f"Доля признаков с дрифтом: {drift_share:.2%}")

    if has_drift:
        print("\n⚠️  Обнаружен дрифт данных - рекомендуется переобучение модели")
    else:
        print("\n✅ Дрифт не обнаружен - модель работает стабильно")

    result.save_html("test_results.html")

    return results_dict


def log_drift_to_mlflow(reference_data, current_data, drift_report):
    """Логирование результатов дрифта в MLflow"""

    mlflow.set_experiment("iris-drift-monitoring")

    with mlflow.start_run(run_name="drift_check"):

        # Извлечение метрик из результатов
        for metric in drift_report.get('metrics', []):
            if metric.get('metric') == 'DriftedColumnsCount':
                result = metric.get('result', {})

                drift_share = result.get('drift_share', 0)
                n_drifted = result.get('number_of_drifted_columns', 0)
                dataset_drift = result.get('dataset_drift', False)

                mlflow.log_metric("drift_share", drift_share)
                mlflow.log_metric("n_drifted_features", n_drifted)
                mlflow.log_metric("dataset_drift", int(dataset_drift))

                print(f"\n📊 Метрики дрифта:")
                print(f"  - Доля дрифта: {drift_share:.2%}")
                print(f"  - Признаков с дрифтом: {n_drifted}")
                break

        # Параметры датасетов
        mlflow.log_param("reference_size", len(reference_data))
        mlflow.log_param("current_size", len(current_data))

        # Логирование HTML отчетов
        import glob
        for html_file in glob.glob("*.html"):
            mlflow.log_artifact(html_file)

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Результаты залогированы в MLflow")
        print(f"   Run ID: {run_id}")


if __name__ == "__main__":
    print("=== 🔬 Анализ Data Drift с Evidently ===\n")

    # Загрузка данных
    iris_data = load_iris_data()
    reference_data, _ = train_test_split(iris_data, test_size=0.3, random_state=42)

    print(f"📁 Референсные данные: {len(reference_data)} образцов\n")

    # Сценарий 1: Данные без дрифта
    print("=" * 60)
    print("Тест 1: Данные без дрифта")
    print("=" * 60)

    current_data_no_drift, _ = train_test_split(iris_data, test_size=0.3, random_state=123)

    report_dict = generate_drift_report(
        reference_data,
        current_data_no_drift,
        "report_no_drift.html"
    )

    test_results = run_drift_tests(reference_data, current_data_no_drift)
    log_drift_to_mlflow(reference_data, current_data_no_drift, test_results)

    # Сценарий 2: Данные с дрифтом признаков
    print("\n" + "=" * 60)
    print("Тест 2: Дрифт признаков (distribution shift)")
    print("=" * 60)

    current_data_shift = simulate_drift_data(reference_data, drift_type='shift')

    report_dict = generate_drift_report(
        reference_data,
        current_data_shift,
        "report_feature_drift.html"
    )

    test_results = run_drift_tests(reference_data, current_data_shift)
    log_drift_to_mlflow(reference_data, current_data_shift, test_results)

    # Сценарий 3: Данные с пропущенными значениями
    print("\n" + "=" * 60)
    print("Тест 3: Проблемы качества данных (missing values)")
    print("=" * 60)

    current_data_missing = simulate_drift_data(reference_data, drift_type='missing')

    report_dict = generate_drift_report(
        reference_data,
        current_data_missing,
        "report_data_quality.html"
    )

    test_results = run_drift_tests(reference_data, current_data_missing)
    log_drift_to_mlflow(reference_data, current_data_missing, test_results)

    # Сценарий 4: Дрифт целевой переменной
    print("\n" + "=" * 60)
    print("Тест 4: Target Drift")
    print("=" * 60)

    current_data_target = simulate_drift_data(reference_data, drift_type='target')

    report_dict = generate_drift_report(
        reference_data,
        current_data_target,
        "report_target_drift.html"
    )

    test_results = run_drift_tests(reference_data, current_data_target)
    log_drift_to_mlflow(reference_data, current_data_target, test_results)

    print("\n" + "=" * 60)
    print("✅ Анализ завершен")
    print("=" * 60)
    print(f"📊 HTML отчеты сохранены в текущей директории")
    print(f"🌐 MLflow UI: https://mlflow.labs.itmo.loc")
    print(f"📁 Эксперимент: iris-drift-monitoring")
