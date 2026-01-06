import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_dataset():
    print("Loading dataset...")
    df = pd.read_csv(  #GIVE THE DATASET PATH HERE
                     )
    print("Dataset loaded | Shape:", df.shape)
    return df

def clean_dataset(df):
    df = df.copy()

    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    df.drop_duplicates(inplace=True)

    # Winsorization for outliers
    for col in df.select_dtypes(include=np.number).columns:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)

    return df

def run_eda(df):
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # -------- Graph 1: Target Distribution --------
    sns.countplot(
        x="default.payment.next.month",
        data=df,
        ax=axes[0]
    )
    axes[0].set_title("Default vs Non-Default Distribution")
    axes[0].set_xlabel("Default (0 = No, 1 = Yes)")
    axes[0].set_ylabel("Count")

    # -------- Graph 2: Correlation Heatmap --------
    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        center=0,
        ax=axes[1]
    )
    axes[1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()

def prepare_features(df):
    X = df.drop("default.payment.next.month", axis=1)
    y = df["default.payment.next.month"]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("\n--- LOGISTIC REGRESSION RESULTS ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probas))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    return model

def train_random_forest(X_train, X_test, y_train, y_test):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("\n--- RANDOM FOREST RESULTS ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probas))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    return model

def train_neural_network(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation="relu", input_dim=X_train_scaled.shape[1]),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2
    )

    preds = (model.predict(X_test_scaled) > 0.5).astype(int)

    print("\n--- DEEP LEARNING RESULTS ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    return model, scaler

def save_models(lr_model, rf_model, nn_model, nn_scaler):
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(lr_model, "artifacts/logistic_model.pkl")
    joblib.dump(rf_model, "artifacts/random_forest_model.pkl")
    joblib.dump(nn_scaler, "artifacts/nn_scaler.pkl")
    nn_model.save("artifacts/neural_network_model.h5")

    print("\nModels saved in 'artifacts/' directory")

def main():
    df = load_dataset()
    df = clean_dataset(df)

    run_eda(df)

    X_train, X_test, y_train, y_test = prepare_features(df)

    lr_model = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    nn_model, nn_scaler = train_neural_network(X_train, X_test, y_train, y_test)

    save_models(lr_model, rf_model, nn_model, nn_scaler)

    print("\nCREDIT DEFAULT RISK ANALYSIS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
