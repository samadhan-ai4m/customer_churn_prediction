import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split




def load_and_split_data():
    data = pd.read_csv(r'data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

    X = data.drop(columns=["Churn", "customerID"])
    y = data["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test 

