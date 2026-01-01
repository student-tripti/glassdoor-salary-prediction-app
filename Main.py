import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("glassdoor_jobs.csv")

# -----------------------------
# Salary cleaning
# -----------------------------
df = df[df["Salary Estimate"] != "-1"]

df["Salary Estimate"] = (
    df["Salary Estimate"]
    .str.replace("$", "", regex=False)
    .str.replace("K", "", regex=False)
    .str.split("-")
)

df["min_salary"] = df["Salary Estimate"].apply(lambda x: int(x[0]))
df["max_salary"] = df["Salary Estimate"].apply(lambda x: int(x[1].split()[0]))
df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2

# -----------------------------
# Features & Target
# -----------------------------
features = [
    "Job Title",
    "Location",
    "Industry",
    "Sector",
    "Type of ownership",
    "Rating",
    "Founded"
]

X = df[features]
y = df["avg_salary"]

# -----------------------------
# Column separation
# -----------------------------
num_features = ["Rating", "Founded"]
cat_features = [
    "Job Title",
    "Location",
    "Industry",
    "Sector",
    "Type of ownership"
]

# -----------------------------
# Pipelines
# -----------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -----------------------------
# Final Pipeline
# -----------------------------
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", Ridge(alpha=1.0))
])

# -----------------------------
# Train
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)

# -----------------------------
# Save pipeline
# -----------------------------
with open("salary_prediction_pipeline.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("✅ Salary prediction pipeline trained & saved successfully!")