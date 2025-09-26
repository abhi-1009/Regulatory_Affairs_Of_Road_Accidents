"""
Steps:
1. Load & clean CSV
2. Exploratory Data Analysis (tables + matplotlib charts)
3. Save cleaned data to MySQL (via SQLAlchemy)
4. Run analytical SQL queries on MySQL
5. Train ML models (LinearRegression & RandomForest) for chosen outcome
6. Export results to Excel workbook
"""

import os, re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------ CONFIG ------------------
CSV_PATH = Path("C:/Users/Hp/OneDrive/Desktop/python/Road_Accident_Analysis/Regulatory Affairs of Road Accident Data 2020 India.csv")

# MySQL connection

MYSQL_CONN = "mysql+pymysql://root:Abhi%40100982@localhost/Road_accident"

TABLE_NAME = "accidents_2020"
EXCEL_PATH = Path("Accident_Insights_2020.xlsx")

ML_TARGET_OUTCOME = "Persons Killed"
RANDOM_STATE = 42
RF_ESTIMATORS = 200
CROSS_VAL_FOLDS = 5

def clean_column_name(c: str) -> str:
    c = str(c).strip().replace("\n", " ")
    c = re.sub(r"\s+", " ", c)
    c = c.lower().strip().replace(" ", "_")
    return c

def strip_text(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def find_best_columns(cols):
    expected = {
        "million_plus_cities": ["million_plus_cities", "million_plus_city", "city", "cities"],
        "cause_category": ["cause_category", "cause"],
        "cause_subcategory": ["cause_subcategory", "cause_sub_category", "subcategory"],
        "outcome_of_incident": ["outcome_of_incident", "outcome", "outcome_of_incidents"],
        "count": ["count", "number", "value", "total"]
    }
    cols_set = set(cols)
    mapping = {}
    for canon, candidates in expected.items():
        for cand in candidates:
            if cand in cols_set:
                mapping[canon] = cand
                break
    return mapping

def main():
    start = datetime.now()
    print(f"Run started: {start}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    # 1) Load CSV
    df_raw = pd.read_csv(CSV_PATH)
    print("Raw shape:", df_raw.shape)
    df_raw.columns = [clean_column_name(c) for c in df_raw.columns]

    # 2) Map columns
    col_map = find_best_columns(df_raw.columns)
    required = ["million_plus_cities","cause_category","cause_subcategory","outcome_of_incident","count"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df_raw.columns)}")

    df = df_raw.rename(columns={col_map[k]: k for k in col_map})
    for c in ["million_plus_cities","cause_category","cause_subcategory","outcome_of_incident"]:
        df[c] = df[c].map(strip_text)

    df["outcome_of_incident"] = df["outcome_of_incident"].replace(
        {"Greviously Injured": "Grievously Injured", "greviously injured": "Grievously Injured"}
    )

    df["count"] = df["count"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    before = df.shape[0]
    df = df.dropna(subset=required).reset_index(drop=True)
    after = df.shape[0]
    print(f"Cleaned data: dropped {before-after}, remaining {after}")

    # 3) EDA
    city_counts = df.groupby("million_plus_cities")["count"].sum().sort_values(ascending=False)
    outcome_counts = df.groupby("outcome_of_incident")["count"].sum().sort_values(ascending=False)
    cause_counts = df.groupby("cause_category")["count"].sum().sort_values(ascending=False)

    print("\nTop 10 Cities:\n", city_counts.head(10))
    print("\nTop Outcomes:\n", outcome_counts.head(10))
    print("\nTop Causes:\n", cause_counts.head(10))

    # 4) Charts
    try:
        plt.figure(figsize=(8,5))
        city_counts.head(10).iloc[::-1].plot(kind="barh")
        plt.title("Top 10 Cities by Total Count")
        plt.show()

        plt.figure(figsize=(10,4))
        outcome_counts.plot(kind="bar")
        plt.title("Total Count by Outcome")
        plt.xticks(rotation=45, ha="right")
        plt.show()

        plt.figure(figsize=(10,4))
        cause_counts.plot(kind="bar")
        plt.title("Total Count by Cause Category")
        plt.xticks(rotation=45, ha="right")
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)

    # 5) SQLAlchemy -> MySQL
    engine = create_engine(MYSQL_CONN, echo=False, future=True)

    df.to_sql(TABLE_NAME, engine, index=False, if_exists="replace")
    print("Data loaded into MySQL table 'accidents_2020'")

    q_outcomes = text("""
        SELECT outcome_of_incident, SUM(count) AS total_count
        FROM accidents_2020
        GROUP BY outcome_of_incident
        ORDER BY total_count DESC;
    """)

    q_top_cities = text("""
        SELECT million_plus_cities, SUM(count) AS total_count
        FROM accidents_2020
        WHERE outcome_of_incident = :target
        GROUP BY million_plus_cities
        ORDER BY total_count DESC
        LIMIT 10;
    """)

    q_causes = text("""
        SELECT cause_category, cause_subcategory, SUM(count) AS total_count
        FROM accidents_2020
        WHERE outcome_of_incident = :target
        GROUP BY cause_category, cause_subcategory
        ORDER BY total_count DESC
        LIMIT 20;
    """)

    with engine.connect() as conn:
        df_q_outcomes = pd.read_sql(q_outcomes, conn)
        df_q_top_cities = pd.read_sql(q_top_cities, conn, params={"target": ML_TARGET_OUTCOME})
        df_q_causes = pd.read_sql(q_causes, conn, params={"target": ML_TARGET_OUTCOME})

    print("\nSQL Results:")
    print(df_q_outcomes.head())
    print(df_q_top_cities.head())
    print(df_q_causes.head())

    # 6) Machine Learning

    df_ml = df[df["outcome_of_incident"] == ML_TARGET_OUTCOME].copy()
    if df_ml.shape[0] >= 10:
        X = df_ml[["million_plus_cities","cause_category","cause_subcategory"]]
        y = df_ml["count"].astype(float)

        # Fixed OneHotEncoder for scikit-learn ≥ 1.2
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              ["million_plus_cities","cause_category","cause_subcategory"])],
            remainder="drop"
        )

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=RF_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        }

        results = {}
        for name, model in models.items():
            pipe = Pipeline([("pre", pre), ("model", model)])
            scores = cross_val_score(
                pipe, X, y,
                cv=min(CROSS_VAL_FOLDS, max(2, df_ml.shape[0] // 2)),
                scoring="neg_mean_absolute_error",
                n_jobs=1   
            )
            results[name] = {
                "cv_mae_mean": -scores.mean(),
                "cv_mae_std": scores.std()
            }

        print("\nML Results:")
        print(pd.DataFrame(results).T)

        # Train best model
        best_name = min(results, key=lambda k: results[k]["cv_mae_mean"])
        pipe_best = Pipeline([("pre", pre), ("model", models[best_name])])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        pipe_best.fit(X_train, y_train)
        preds = pipe_best.predict(X_test)

        print(
            f"\nBest model: {best_name}, "
            f"MAE: {mean_absolute_error(y_test, preds):.2f}, "
            f"R2: {r2_score(y_test, preds):.3f}"
        )
    else:
        print("Not enough rows for ML.")

    # 7) Excel export
    summary_sheets = {
        "top_cities_total": city_counts.to_frame("total_count"),
        "outcome_totals": outcome_counts.to_frame("total_count"),
        "cause_category_totals": cause_counts.to_frame("total_count"),
        "sql_outcome_totals": df_q_outcomes,
        "sql_top_cities_target": df_q_top_cities,
        "sql_causes_target": df_q_causes,
        "cleaned_sample": df.sample(min(500, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    }

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet, data in summary_sheets.items():
            safe_name = sheet[:31]
            data.to_excel(writer, sheet_name=safe_name, index=True)
    print("\nExcel exported to:", EXCEL_PATH.resolve())

    end = datetime.now()
    print(f"Run finished: {end} (duration {end-start})")

if __name__ == "__main__":
    main()

# Streamlit 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# STREAMLIT CONFIG 

st.set_page_config(page_title="India Road Accident Analysis 2020", layout="wide")

# CONFIG
MYSQL_CONN = "mysql+pymysql://root:Abhi%40100982@localhost/Road_accident"
TABLE_NAME = "accidents_2020"

# DB Connection
@st.cache_data
def load_from_mysql():
    engine = create_engine(MYSQL_CONN, echo=False, future=True)
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    return df

df = load_from_mysql()

# CSV UPLOAD HANDLING
st.sidebar.header("Dataset Options")
upload = st.sidebar.file_uploader("Upload a CSV file (optional)", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.sidebar.success("Using uploaded dataset")
else:
    df = load_from_mysql()
    st.sidebar.info("Using default MySQL dataset")

# Streamlit UI

st.title("Road Accident Data Analysis (India - 2020)")
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Overview", "SQL Queries", "Visualization", "Machine Learning"])

# Overview Section
if section == "Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Cities", df["million_plus_cities"].nunique())
    col3.metric("Unique Outcomes", df["outcome_of_incident"].nunique())

# SQL Queries Section

elif section == "SQL Queries":
    if upload is not None:
        st.warning("SQL Queries are disabled when using an uploaded CSV. Switch to default MySQL dataset.")
    else:
        st.subheader("Run SQL Queries")

    query_type = st.selectbox("Select Query", [
        "Total by Outcome",
        "Top 10 Cities (for Outcome)",
        "Top 20 Causes (for Outcome)"
    ])

    engine = create_engine(MYSQL_CONN, echo=False, future=True)

    if query_type == "Total by Outcome":
        sql = text("""
            SELECT outcome_of_incident, SUM(count) AS total_count
            FROM accidents_2020
            GROUP BY outcome_of_incident
            ORDER BY total_count DESC;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)

    elif query_type == "Top 10 Cities (for Outcome)":
        target = st.selectbox("Select Outcome", df["outcome_of_incident"].unique())
        sql = text("""
            SELECT million_plus_cities, SUM(count) AS total_count
            FROM accidents_2020
            WHERE outcome_of_incident = :target
            GROUP BY million_plus_cities
            ORDER BY total_count DESC
            LIMIT 10;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn, params={"target": target})
        st.write(results)

    elif query_type == "Top 20 Causes (for Outcome)":
        target = st.selectbox("Select Outcome", df["outcome_of_incident"].unique())
        sql = text("""
            SELECT cause_category, cause_subcategory, SUM(count) AS total_count
            FROM accidents_2020
            WHERE outcome_of_incident = :target
            GROUP BY cause_category, cause_subcategory
            ORDER BY total_count DESC
            LIMIT 20;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn, params={"target": target})
        st.write(results)

# -------------------------
# Visualization Section
# -------------------------
elif section == "Visualization":
    st.subheader("Charts & Visualizations")

    choice = st.selectbox("Choose Visualization", [
        "Top 10 Cities",
        "Total by Outcome",
        "Top Causes"
    ])

    if choice == "Top 10 Cities":
        city_counts = df.groupby("million_plus_cities")["count"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(city_counts)

    elif choice == "Total by Outcome":
        outcome_counts = df.groupby("outcome_of_incident")["count"].sum().sort_values(ascending=False)
        st.bar_chart(outcome_counts)

    elif choice == "Top Causes":
        cause_counts = df.groupby("cause_category")["count"].sum().sort_values(ascending=False)
        st.bar_chart(cause_counts)

# -------------------------
# Machine Learning Section
# -------------------------
elif section == "Machine Learning":
    st.subheader("ML Model for Predicting Counts")

    target = st.selectbox("Select Outcome for Prediction", df["outcome_of_incident"].unique())
    df_ml = df[df["outcome_of_incident"] == target].copy()

    if df_ml.shape[0] < 10:
        st.warning("Not enough data for ML on this outcome")
    else:
        X = df_ml[["million_plus_cities","cause_category","cause_subcategory"]]
        y = df_ml["count"].astype(float)

        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              ["million_plus_cities","cause_category","cause_subcategory"])],
            remainder="drop"
        )


        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        }

        for name, model in models.items():
            pipe = Pipeline([("pre", pre), ("model", model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            st.markdown(f"**{name}**")
            st.write(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
            st.write(f"R²: {r2_score(y_test, preds):.3f}")
            st.divider()
