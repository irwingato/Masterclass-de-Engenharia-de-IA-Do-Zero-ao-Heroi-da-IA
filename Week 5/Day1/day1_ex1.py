#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Define features and target
features = df[['total_bill', 'size']]
target = df['tip']

print("Features: \n", features.head())
print("Target: \n", target.head())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training Data Set shape: ", X_train.shape)
print("Test Data Set shape: ", X_test.shape)

# Visualize relationships
g = sns.pairplot(df, x_vars=["total_bill", "size"], y_vars="tip", height=5, aspect=0.8, kind="reg")
g.fig.suptitle("Feature vs Target Relation", y=1.02)
plt.show()