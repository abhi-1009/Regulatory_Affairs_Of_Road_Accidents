# Regulatory_Affairs_Of_Road_Accidents
This project analyzes road accident data across 50 million-plus cities in India (2020)

It leverages Python, SQL, Machine Learning, and Streamlit to:
- Clean and preprocess accident data
- Explore causes, outcomes, and accident patterns
- Run SQL queries for deeper insights
- Train ML models to predict accident severity (Counts of "Persons Killed")
- Deploy an interactive Streamlit app for data exploration

# Dataset
Source: [data.gov.in – Road Accidents in India 2020](https://data.gov.in/catalog/road-accidents-india-2020)  

Columns:
1. `Million Plus Cities` → City names  
2. `Cause Category` → Broad categories (Traffic Control, Junction, Road Features, etc.)  
3. `Cause Subcategory` → Detailed cause description  
4. `Outcome of Incident` → e.g., Persons Killed, Minor Injury, Total Accidents  
5. `Count` → Number of incidents  

# Tools & Technologies
- **Python** → Data preprocessing, analysis & visualization  
- **Pandas, Matplotlib, Seaborn** → Data wrangling & EDA charts  
- **MySQL + SQLAlchemy** → Storing & querying accident data  
- **Scikit-learn** → Machine Learning models (Linear Regression & Random Forest)  
- **Streamlit** → Interactive web application  
- **Excel (openpyxl)** → Exporting insights  

# Workflow
1. **Data Preprocessing**  
   - Clean column names & missing values  
   - Convert `Count` column to numeric  
   - Standardize labels (e.g., “Greviously Injured” → “Grievously Injured”)  
   - Export cleaned dataset  

2. **Exploratory Data Analysis (EDA)**  
   - Top 10 accident-prone cities  
   - Distribution of causes & outcomes  
   - Accident Causes vs Outcomes  

3. **SQL Queries & Insights**  
   - Total by outcome  
   - Top 10 cities (Persons Killed)  
   - Top 20 causes (Persons Killed)  

4. **Machine Learning**  
   - Train models to predict **Counts of Persons Killed**  
   - Compare **Linear Regression vs Random Forest**  
   - Evaluate using **MAE and R²**  

5. **Streamlit App**  
   - Overview (dataset preview & summary metrics)  
   - SQL Query Runner  
   - Visualizations  
   - Machine Learning predictions

# Key Insights
   Delhi, Bengaluru, and Chennai recorded the highest accident counts.
   Traffic Control & Road Features were leading causes of accidents.
   Random Forest outperformed Linear Regression with lower MAE and higher R².

# Future Enhancements
   Add time-series forecasting for accident trends
   Include geospatial analysis with maps.
