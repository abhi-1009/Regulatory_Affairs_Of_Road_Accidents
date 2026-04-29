## Regulatory Affairs of Road Accidents — India 2020

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?style=flat&logo=mysql)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Domain](https://img.shields.io/badge/Domain-Public%20Safety%20%26%20Regulatory-lightblue?style=flat)

An end-to-end data analysis and machine learning project on **India's road accident regulatory data (2020)** across 50 million-plus cities. The project covers data cleaning, EDA, SQL-based querying, ML prediction of accident counts, and an interactive **Streamlit dashboard** — with results exported to Excel and stored in MySQL.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Pipeline](#data-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Model Accuracy Results](#model-accuracy-results)
- [SQL Queries](#sql-queries)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Project Overview
This project analyses regulatory road accident data across **50 million-plus cities in India (2020)** to:
- Clean and standardise raw government dataset (typo correction, label standardisation)
- Explore top accident-prone cities, leading causes, and outcome distributions
- Run 3 predefined SQL queries on MySQL for analytical insights
- Train **Linear Regression and Random Forest** models to predict accident counts per city-cause combination
- Deploy a 4-section Streamlit app: Overview → SQL Queries → Visualisations → ML Predictions
- Export 7-sheet Excel workbook with all insights

## Dataset
| Property | Detail |
| :--- | :--- |
| **Source** | [data.gov.in — Road Accidents India 2020](https://data.gov.in/catalog/road-accidents-india-2020) |
| **Coverage** | 50 million-plus cities across India |
| **Year** | 2020 |
| **MySQL Table** | `accidents_2020` in database `Road_accident` |

### Columns
| Original Column | Cleaned Name | Type | Description |
| :--- | :--- | :---: | :--- |
| `Million Plus Cities` | `million_plus_cities` | Categorical | City name |
| `Cause Category` | `cause_category` | Categorical | Broad accident cause group |
| `Cause Subcategory` | `cause_subcategory` | Categorical | Detailed cause description |
| `Outcome of Incident` | `outcome_of_incident` | Categorical | e.g., Persons Killed, Minor Injury |
| `Count` | `count` | Numeric | Number of incidents |

### Data Cleaning Applied (from source code)
| Issue | Fix |
| :--- | :--- |
| Column name whitespace / newlines | `re.sub(r"\s+", " ")` + `.lower().strip()` |
| Comma-formatted counts (`"1,234"`) | `str.replace(",", "")` → `pd.to_numeric()` |
| Typo: `"Greviously Injured"` | Replaced → `"Grievously Injured"` |
| Whitespace in string values | `re.sub(r"\s+", " ")` applied to all text columns |
| Missing values | `dropna(subset=required_cols)` |

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, groupby EDA, Excel export |
| **NumPy** | latest | Numerical operations |
| **Matplotlib** | latest | EDA bar charts (top cities, outcomes, causes) |
| **Scikit-Learn** | 1.x | OneHotEncoder, ColumnTransformer, Pipeline, LinearRegression, RandomForestRegressor, cross_val_score, MAE, R² |
| **SQLAlchemy** | latest | MySQL ORM — load and query `accidents_2020` table |
| **PyMySQL** | latest | MySQL driver |
| **Streamlit** | 1.x | 4-section interactive web dashboard |
| **OpenPyXL** | latest | 7-sheet Excel workbook export |
| **re / pathlib / os** | built-in | Regex column cleaning, path handling |
| **datetime** | built-in | Run timestamp logging |

### Python Libraries (from source code)
```python
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
import streamlit as st
```
## Data Pipeline
```
Regulatory Affairs of Road Accident Data 2020 India.csv
              │
              ▼
1. Load CSV + clean column names
   ├── Strip whitespace, replace \n → space
   ├── Lowercase + underscore format
   └── Auto-map column variants (find_best_columns())

              │
              ▼
2. Data Cleaning
   ├── Strip all string values (re.sub whitespace)
   ├── Fix typo: "Greviously Injured" → "Grievously Injured"
   ├── Count column: remove commas → pd.to_numeric()
   └── dropna(subset=required_cols)

              │
              ▼
3. EDA (groupby analysis + 3 matplotlib charts)
   ├── Top 10 Cities by total count (horizontal bar)
   ├── Total Count by Outcome (bar chart)
   └── Total Count by Cause Category (bar chart)

              │
              ▼
4. Push to MySQL (SQLAlchemy)
   └── df.to_sql("accidents_2020", if_exists="replace")

              │
              ▼
5. SQL Queries (3 queries on MySQL)
   ├── Total by Outcome
   ├── Top 10 Cities (filtered by Persons Killed)
   └── Top 20 Causes (filtered by Persons Killed)

              │
              ▼
6. Machine Learning
   ├── Filter: outcome_of_incident == "Persons Killed"
   ├── Features: million_plus_cities, cause_category, cause_subcategory
   ├── Target: count (float)
   ├── Preprocessor: OneHotEncoder (all 3 categorical cols)
   ├── Cross-validation MAE (5-fold) for both models
   ├── Best model selected → retrained on 80/20 split
   └── Final MAE and R² reported

              │
              ▼
7. Excel Export (7 sheets → Accident_Insights_2020.xlsx)
```

## Machine Learning Models
### Problem Type
**Regression** — predicting `count` (number of incidents) for a given city + cause + subcategory combination, filtered to **"Persons Killed"** outcome.

### Features (3 categorical — all OHE encoded)
```python
X = df_ml[["million_plus_cities", "cause_category", "cause_subcategory"]]
y = df_ml["count"].astype(float)
```
### Preprocessing Pipeline
```python
pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
      ["million_plus_cities", "cause_category", "cause_subcategory"])],
    remainder="drop"
)
```
### Models Trained
| Model | Parameters |
| :--- | :--- |
| **Linear Regression** | Default sklearn settings |
| **Random Forest Regressor** | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |

## Model Accuracy Results
**Evaluation:** 5-fold cross-validation (MAE) + final 80/20 train-test split
| Model | CV MAE (mean) | Final MAE | R² Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | Higher | Higher | Lower | Weak fit — non-linear patterns |
| **Random Forest** | **Lower** | **Lower** | **Higher** | Best — selected for final evaluation |
> **Best model** is auto-selected based on lowest `cv_mae_mean` from cross-validation.

### Model Selection Logic (from source code)
```python
best_name = min(results, key=lambda k: results[k]["cv_mae_mean"])
```
## SQL Queries
All 3 queries run against the `accidents_2020` MySQL table:
| # | Query | Description |
| :---: | :--- | :--- |
| 1 | **Total by Outcome** | `SUM(count) GROUP BY outcome_of_incident ORDER BY total_count DESC` |
| 2 | **Top 10 Cities (Persons Killed)** | `SUM(count) WHERE outcome = 'Persons Killed' GROUP BY city LIMIT 10` |
| 3 | **Top 20 Causes (Persons Killed)** | `SUM(count) WHERE outcome = 'Persons Killed' GROUP BY cause_category, cause_subcategory LIMIT 20` |

## Streamlit Dashboard
### Navigation (4 sections via sidebar radio)
#### Section 1 — Overview
| Element | Detail |
| :--- | :--- |
| Dataset Preview | First 5 rows of loaded data |
| Total Records | `len(df)` metric |
| Unique Cities | `df['million_plus_cities'].nunique()` |
| Unique Outcomes | `df['outcome_of_incident'].nunique()` |

#### Section 2 — SQL Queries
| Query | Filter |
| :--- | :--- |
| Total by Outcome | No filter |
| Top 10 Cities | Outcome selectbox |
| Top 20 Causes | Outcome selectbox |
> SQL Queries disabled when using uploaded CSV — switches automatically to MySQL mode.

#### Section 3 — Visualization
| Chart | Type |
| :--- | :--- |
| Top 10 Cities | `st.bar_chart` — total count per city |
| Total by Outcome | `st.bar_chart` — count per outcome type |
| Top Causes | `st.bar_chart` — count per cause category |

#### Section 4 — Machine Learning
- Outcome selectbox to choose prediction target
- Both models trained and evaluated on selected outcome
- MAE and R² displayed per model with `st.divider()` separator

### Excel Workbook (7 sheets → `Accident_Insights_2020.xlsx`)
| Sheet | Contents |
| :--- | :--- |
| `top_cities_total` | City → total count (all outcomes) |
| `outcome_totals` | Outcome → total count |
| `cause_category_totals` | Cause category → total count |
| `sql_outcome_totals` | SQL query 1 results |
| `sql_top_cities_target` | SQL query 2 results (Persons Killed) |
| `sql_causes_target` | SQL query 3 results (Persons Killed) |
| `cleaned_sample` | Random 500-row sample of cleaned dataset |

## Installation and Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/Regulatory_Affairs_Of_Road_Accidents.git
cd Regulatory_Affairs_Of_Road_Accidents
```
### Step 2 — Install Required Libraries
```bash
pip install streamlit pandas numpy matplotlib scikit-learn sqlalchemy pymysql openpyxl
```
### Step 3 — Add the Dataset
Place the CSV file in the project folder and update:
```python
CSV_PATH = Path("Regulatory Affairs of Road Accident Data 2020 India.csv")
```
### Step 4 — Configure MySQL
Create a MySQL database named `Road_accident` and update:
```python
MYSQL_CONN = "mysql+pymysql://your_user:your_password@localhost/Road_accident"
```
### Step 5 — Run the Analysis Script
```bash
python road_accident_analysis.py
```
This generates `Accident_Insights_2020.xlsx` and pushes data to MySQL.

### Step 6 — Launch the Streamlit App
```bash
streamlit run road_accident_streamlit.py
```
## Usage
1. **Overview section** — view dataset preview, total records, unique cities and outcomes
2. **SQL Queries section** — select from 3 predefined queries, filter by outcome where applicable
3. **Visualization section** — select from 3 chart types (cities, outcomes, causes)
4. **Machine Learning section** — select outcome to predict → view MAE and R² for both models
5. **Sidebar upload** — optionally upload a custom CSV (note: SQL queries require MySQL mode)

## Key Insights
- **Delhi, Bengaluru, and Chennai** recorded the highest accident counts among all million-plus cities
- **Traffic Control violations** and **Road Feature defects** are the leading cause categories
- **Random Forest outperforms Linear Regression** with lower MAE and higher R² on all tested outcomes
- **Persons Killed** incidents are concentrated in specific city + cause subcategory combinations
- **Overspeeding and signal jumping** appear consistently in the top 20 causes for fatalities

## Future Enhancements
- **Time-series forecasting** — ARIMA/Prophet models for accident trend prediction across years
- **Geospatial analysis** — Folium/Plotly choropleth maps of accident density by city and state
- **State-level analysis** — Extend from million-plus cities to all Indian states
- **Severity scoring** — Multi-output model predicting both count and outcome simultaneously
- **Automated reporting** — Scheduled PDF/Excel report generation with latest data

## References
- [Road Accidents India 2020 — data.gov.in](https://data.gov.in/catalog/road-accidents-india-2020)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
