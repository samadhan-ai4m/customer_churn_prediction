
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from preprocessing import load_and_split_data
from pipeline import create_preprocessor



X_train, X_test, y_train, y_test = load_and_split_data()


preprocessor = create_preprocessor(X_train)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training Logistic Regression model...")

model.fit(X_train, y_train)

print("Training completed!")

y_pred = model.predict(X_test)


print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "models/churn_model.pkl")
print("\nModel saved successfully at models/churn_model.pkl")
