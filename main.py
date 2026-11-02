
import re
import joblib
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# Один раз скачать данные для nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Загрузка данных
df = pd.read_excel("data/five treats.xlsx")  # или .csv

print("Размер датасета:", df.shape)
print("Колонки:", df.columns.tolist())
print(df.head(3))

# Предобработка текста

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # нижний регистр
    text = str(text).lower()
    # убрать пунктуацию и цифры
    text = re.sub(r'[^a-z\s]', '', text)
    # разделить слова на массив
    words = text.split()
    # убрать стоп-слова
    words = [w for w in words if w not in stop_words]
    # лемматизация
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df['clean_text'] = df['context'].apply(preprocess)

print("\nПример до/после:")
print("ДО:  ", df['context'].iloc[0])
print("ПОСЛЕ:", df['clean_text'].iloc[0])

# Создаём пять бинарных колонок из мультиклассовой разметки
df['Openness'] = (df['five treats'] == 0).astype(int)
df['Conscientiousness'] = (df['five treats'] == 1).astype(int)
df['Extraversion'] = (df['five treats'] == 2).astype(int)
df['Agreeableness'] = (df['five treats'] == 3).astype(int)
df['Emotional_Stability'] = (df['five treats'] == 4).astype(int)

# Проверяем, что получилось
print("Новые колонки:", df.columns.tolist())
print(df[['five treats', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional_Stability']].head(10))

# Функция для обучения и оценки одной модели

def train_and_evaluate(texts, labels, model_name, classifier):
    # 1. Разделение на train/test — ЗДЕСЬ создаются X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 2. Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', classifier)
    ])

    # 3. Обучение
    pipeline.fit(X_train, y_train)

    # 4. Предсказание
    y_pred = pipeline.predict(X_test)

    # 5. Метрики
    print(classification_report(y_test, y_pred, zero_division=0))
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return pipeline, f1


# Обучение моделей по каждой черте

traits = [
    'Openness',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Emotional_Stability'
]
results = {}   # сюда сложим лучшие модели и метрики

for trait in traits:
    print(f"# ЧЕРТА: {trait}")

    # Проверяем, что колонка существует
    if trait not in df.columns:
        print(f" Колонка '{trait}' не найдена. Пропускаем.")
        continue

    X = df['clean_text']
    y = df[trait]

    # Logistic Regression
    lr_pipeline, lr_f1 = train_and_evaluate(
        X, y, "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

    # Random Forest
    rf_pipeline, rf_f1 = train_and_evaluate(
        X, y, "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42)
    )

    # Выбираем лучшую модель для этой черты
    if lr_f1 >= rf_f1:
        best_model, best_name, best_f1 = lr_pipeline, "Logistic Regression", lr_f1
    else:
        best_model, best_name, best_f1 = rf_pipeline, "Random Forest", rf_f1

    results[trait] = {
        'model': best_model,
        'name': best_name,
        'f1': best_f1
    }

    print(f"\n Лучшая модель для '{trait}': {best_name} (F1={best_f1:.4f})")

# Итоговая таблица результатов

print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")

summary = pd.DataFrame([
    {'Черта': trait, 'Лучшая модель': res['name'], 'F1 (weighted)': round(res['f1'], 4)}
    for trait, res in results.items()
])
print(summary.to_string(index=False))

# Сохранение лучших моделей

import os
os.makedirs('models', exist_ok=True)

for trait, res in results.items():
    filename = f"models/{trait}_best.joblib"
    joblib.dump(res['model'], filename)
    print(f"Сохранено: {filename}")

print("\n Готово!")