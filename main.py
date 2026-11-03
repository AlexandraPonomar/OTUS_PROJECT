
import re
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display_functions import display

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. НАСТРОЙКИ И ЗАГРУЗКА ДАННЫХ

RANDOM_STATE = 42

TRAIT_MAPPING = {
    0: "Openness",
    1: "Conscientiousness",
    2: "Extraversion",
    3: "Agreeableness",
    4: "Emotional_Stability"
}

TRAITS = list(TRAIT_MAPPING.values())

df = pd.read_excel("data/five treats.xlsx")

df = df.rename(columns={
    "five treats": "label",
    "context": "text"
})

print("Исходный размер:", df.shape)

# 2. ОЧИСТКА ДАННЫХ

print("\nПропущенные значения:")
print(df.isna().sum())

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

# Проверяем неизвестные метки
invalid_rows = df[
    df["label"].notna()
    & ~df["label"].isin(TRAIT_MAPPING)
]

print("\nСтроки с неизвестными метками:")
display(invalid_rows)

# Удаляем пропуски и неизвестные классы
df = df.dropna(subset=["text", "label"]).copy()
df = df[df["label"].isin(TRAIT_MAPPING)].copy()
df["label"] = df["label"].astype(int)

# 3. ПРЕДОБРАБОТКА ТЕКСТА

def clean_text(text):
    text = str(text).lower()

    # Удаление URL
    text = re.sub(
        r"https?://\S+|www\.\S+", " ", text
    )

    # Удаление HTML-тегов
    text = re.sub(r"<[^>]+>", " ", text)

    # Удаление пунктуации и спецсимволов
    text = re.sub(r"[^a-z\s']", " ", text)

    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text_clean"] = df["text"].apply(clean_text)
df = df[df["text_clean"].str.len() > 0].copy()

# 4. СОЗДАНИЕ ПЯТИ БИНАРНЫХ КОЛОНОК

for class_id, trait in TRAIT_MAPPING.items():
    df[trait] = (df["label"] == class_id).astype(int)

assert (df[TRAITS].sum(axis=1) == 1).all()
assert df[TRAITS].isin([0, 1]).all().all()

print("\nПример бинарной разметки:")
display(df[["text_clean"] + TRAITS].head())

# 5. ПРОВЕРКА И УДАЛЕНИЕ ДУБЛИКАТОВ

print("\nКоличество дубликатов:")
print(df["text_clean"].duplicated().sum())

conflicts = (
    df.groupby("text_clean")["label"]
    .nunique()
)

if (conflicts > 1).any():
    raise ValueError(
        "Обнаружены одинаковые тексты с разными метками"
    )

df = df.drop_duplicates(
    subset="text_clean"
).reset_index(drop=True)

print("Размер после очистки:", df.shape)


# 6. АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ

class_distribution = df[TRAITS].sum()

print("\nРаспределение классов:")
display(class_distribution.to_frame("Количество"))

class_distribution.plot(
    kind="bar",
    figsize=(11, 5),
    title="Распределение Big Five"
)

plt.xlabel("Черта личности")
plt.ylabel("Количество предложений")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# 7. РАЗДЕЛЕНИЕ TRAIN / VALIDATION / TEST

if df["label"].value_counts().min() < 5:
    raise ValueError(
        "Слишком мало примеров для стратификации"
    )

train_df, temp_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)

print("\nРазмеры выборок:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# 8. ПОДГОТОВКА ПРИЗНАКОВ И МЕТОК

X_train = train_df["text_clean"]
X_val = val_df["text_clean"]
X_test = test_df["text_clean"]

# Одна целевая колонка с пятью классами
y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# 9. TF-IDF ВЕКТОРИЗАЦИЯ

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

# Обучаем TF-IDF только на Train
X_train_tfidf = tfidf.fit_transform(X_train)

# Применяем к Validation и Test
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("\nРазмеры TF-IDF матриц:")
print("Train:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Test:", X_test_tfidf.shape)

# 10. СОЗДАНИЕ ДВУХ МОДЕЛЕЙ

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE
    )
}

# 11. ОБУЧЕНИЕ И ОЦЕНКА НА VALIDATION

results = []

for name, model in models.items():

    print(f"\nОбучение: {name}")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_val_tfidf)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "F1": f1_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        )
    })

    print(classification_report(
        y_val,
        y_pred,
        labels=list(TRAIT_MAPPING),
        target_names=TRAITS,
        zero_division=0
    ))

# 12. СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\nИтоговое сравнение моделей:")
display(results_df.round(3))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Сравнение моделей Big Five")
plt.ylabel("Значение метрики")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 13. ВЫБОР ЛУЧШЕЙ МОДЕЛИ

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print("\nЛучшая модель:", best_model_name)

# 14. ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ

y_test_pred = best_model.predict(X_test_tfidf)

print("\nРезультаты на Test:")
print(classification_report(
    y_test,
    y_test_pred,
    labels=list(TRAIT_MAPPING),
    target_names=TRAITS,
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    labels=list(TRAIT_MAPPING),
    display_labels=TRAITS,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.show()


# 15. ТАБЛИЦА ПРЕДСКАЗАНИЙ


predictions_df = test_df[
    ["text", "label"]
].copy()

predictions_df["predicted_label"] = y_test_pred

predictions_df["actual_trait"] = (
    predictions_df["label"].map(TRAIT_MAPPING)
)

predictions_df["predicted_trait"] = (
    predictions_df["predicted_label"].map(TRAIT_MAPPING)
)

predictions_df["correct"] = (
    predictions_df["label"]
    == predictions_df["predicted_label"]
)

#Создаём пять бинарных колонок
for class_id, trait in TRAIT_MAPPING.items():
    predictions_df[trait] = (
        predictions_df["predicted_label"] == class_id
    ).astype(int)

#Настраиваем отображение
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)

#Выбираем и упорядочиваем колонки
display_columns = [
    "text",
    "actual_trait",
    "predicted_trait",
    "correct"
] + TRAITS

#Подсвечиваем правильные и ошибочные предсказания
styled_predictions = (
    predictions_df[display_columns]
    .head(20)
    .style
    .map(
        lambda x: (
            "background-color: #d4edda; color: #155724"
            if x else
            "background-color: #f8d7da; color: #721c24"
        ),
        subset=["correct"]
    )
    .format({"text": lambda x: x[:100] + "..." if len(x) > 100 else x})
)

print("\nПримеры предсказаний:")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 200)

display(predictions_df.head(15))

display(styled_predictions)

# 16. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

df.to_csv(
    "five_traits_preprocessed.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    "five_traits_train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    "five_traits_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    "five_traits_test.csv",
    index=False,
    encoding="utf-8-sig"
)

results_df.to_csv(
    "five_traits_model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions_df.to_csv(
    "five_traits_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nОбучение и тестирование завершены!")
