import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/ai4i2020.csv")

print(df.head())
print(df.info())
print(df.describe())

sns.countplot(
    x="Machine failure",
    data=df
)

plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.show()