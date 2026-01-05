import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Student Performance Prediction")

# ---------- LOAD DATA ----------
file_path ="Task_students_performance_dataset.xlsx"
data = pd.read_excel(file_path)

st.subheader("Original Dataset")
st.dataframe(data)

# ---------- TARGET ----------
target_column = "Final_Score"

if target_column not in data.columns:
    st.error("Final_Score column not found")
    st.stop()

# ---------- DROP ID COLUMN (VERY IMPORTANT) ----------
id_columns = ["Student_ID"]
data = data.drop(columns=id_columns, errors="ignore")

# ---------- SPLIT FEATURES & TARGET ----------
X = data.drop(columns=[target_column])
y = data[target_column]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)


# ---------- ENCODE CATEGORICAL FEATURES ----------
X_encoded = pd.get_dummies(X, drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_encoded = scaler.fit_transform(X_encoded)


st.subheader("Encoded Features (ID Removed)")
st.dataframe(X_encoded)

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X, y)


# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- PREDICTION ----------
y_pred = model.predict(X_test)

# ---------- PERFORMANCE ----------
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write("R² Score:", r2)

st.success("Model trained successfully ✅")
