import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder



def clean_data(X):
    X = X.copy()

    X = X.replace(r"^\s*$", float("nan"), regex=True)

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce"
    )

    return X


def create_preprocessor(X_train):

    
    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    
    categorical_columns = [
        col for col in X_train.columns
        if col not in numerical_columns
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

  
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    
    column_transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

  
    preprocessor = Pipeline([
        ("cleaning", FunctionTransformer(clean_data)),
        ("column_transformer", column_transformer)
    ])

    return preprocessor
