import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r'data\WA_Fn-UseC_-Telco-Customer-Churn.csv')


print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.shape)
print(data.nunique())
print(data.describe())


plt.subplot(2, 2, 1)
sns.countplot(data=data, x="Churn")
plt.subplot(2, 2, 2)
sns.countplot(data=data, x="Contract", hue="Churn")
plt.subplot(2, 2, 3)
sns.histplot(
    data=data,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="layer",
    kde=True
)
plt.subplot(2, 2, 4)
sns.boxplot(data=data, x="Churn", y="MonthlyCharges")
plt.show()