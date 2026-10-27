import joblib
import pandas as pd


model = joblib.load("models/lightgbm_pipeline.pkl")


customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 53.85,
    "TotalCharges": 108.15
}


actual_churn = 1

customer_df = pd.DataFrame([customer])


prediction = model.predict(customer_df)[0]


print("Actual Churn:   ", actual_churn)
print("Predicted Churn:", prediction)


