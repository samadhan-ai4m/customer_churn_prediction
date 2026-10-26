import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier

from preprocessing import load_and_split_data
from pipeline import create_preprocessor


X_train, X_test, y_train, y_test = load_and_split_data()


preprocessor = create_preprocessor(X_train)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    ))
])


print("Training LightGBM model...")

model.fit(X_train, y_train)

print("Training completed!")


y_pred = model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("Classification Report")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


joblib.dump(model, "models/lightgbm_pipeline.pkl")

print("LightGBM pipeline saved successfully!")
