# Churn Predictor

End-to-end ML-проект прогнозирования оттока клиентов телеком-оператора. От обучения модели до API-сервиса с интерпретацией и бизнес-метриками.

## Бизнес-задача

Снижение оттока клиентов (churn rate). Модель предсказывает вероятность ухода клиента, чтобы retention-отдел мог предложить превентивные меры до того, как клиент уйдёт.

- Потеря клиента стоит $500
- Удержание стоит $100
- Порог классификации подобран по бизнес-экономике, а не по accuracy

## Данные

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7043 клиента, 21 признак: демография, услуги, платежи, длительность обслуживания.

## Метрики (порог 0.20)

| Метрика | Значение |
|---------|----------|
| Recall | 0.85 |
| Precision | 0.48 |
| F1 | 0.61 |
| Бизнес-потери | $62,400 |

Приоритет — высокий Recall (находим 85% уходящих клиентов).

## Стек

- Python 3.12
- CatBoost
- Scikit-learn (Pipeline, GridSearchCV)
- FastAPI
- SHAP (интерпретация)
- Uvicorn

## Структура проекта

churn-predictor/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   └── threshold.txt
├── src/
│   ├── train.py
│   └── api.py
├── requirements.txt
└── README.md

## Установка и запуск

```bash
git clone https://github.com/karlwoww/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt