import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# 1. Загрузка данных
print("1. Загрузка данных...")
df = pd.read_csv("../data/Telco-Customer-Churn.csv")

# Чистка
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)


# 2. Признаки и целевая
print("2. Подготовка признаков...")
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Пайплайн предобработки
print("3. Создание пайплайна")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(verbose=0, random_seed=42))
])

# 4. Тюнинг гиперпараметров
print("4. Тюнинг гиперпараметров")
param_grid = {
    'classifier__depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Лучшие параметры: {grid.best_params_}")


# 5. Подбор порога по бизнес-метрике
print("5. Подбор порога")
y_proba = best_model.predict_proba(X_test)[:, 1]

cost_of_losing = 500
cost_of_retention = 100

best_threshold = 0.5
best_cost = float('inf')

for t in np.arange(0.2, 0.7, 0.05):
    y_pred = (y_proba >= t).astype(int)
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    total_cost = fn * cost_of_losing + fp * cost_of_retention
    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = t

print(f"Лучший порог: {best_threshold:.2f}")
print(f"Бизнес-потери: ${best_cost:,.0f}")

# 6. Финальные метрики
print("6. Финальные метрики")
y_pred_final = (y_proba >= best_threshold).astype(int)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_final):.4f}")


# 7. Сохранение модели и порога
print("7. Сохранение модели")
joblib.dump(best_model, "../models/churn_model.pkl")
with open("../models/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\nГотово! Модель сохранена в models/churn_model.pkl")
print(f"Порог сохранён в models/threshold.txt: {best_threshold:.2f}")