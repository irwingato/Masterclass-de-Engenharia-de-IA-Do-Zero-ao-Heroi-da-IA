import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Dipslay dataset Info
print("Dataset Info: \n")
print(df.info())

# Preview the first few rows
print("\nFirst 5 Rows: \n")
print(df.head())

# Separete features
categorical_features = df.select_dtypes(include=["object"]).columns
numberical_features = df.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Features: \n", categorical_features.tolist())
print("\nNumerical Features: \n", numberical_features.tolist)

# Display summary of categorical features
print("\n Cateogrical Feature Summary: \n")
for col in categorical_features:
    print(f"{col}: {df[col].value_counts()}\n")
    
# Display Summary of numberical features
print("\n Numerical Feature Summary: \n")
print(df[numberical_features].describe())