
# Rossmann Store Sales Forecasting

## Project Overview

This project focuses on predicting **daily sales for Rossmann stores** using historical sales data and store-level information. The task is a **regression problem**, where the goal is to estimate the numerical value of sales for each store on a given day.

Accurate sales forecasting is important in retail because it supports better decision-making in areas such as inventory planning, staffing, promotions, and financial management. By using machine learning, this project aims to identify the main factors that influence store sales and build models that can generate reliable predictions for unseen data.

---

## Dataset Description

The project uses three files:

- **`train.csv`**: contains historical daily sales records and is used for training and evaluation.
- **`test.csv`**: contains unseen daily records for which sales predictions must be generated.
- **`store.csv`**: contains additional information about each store, such as store type, assortment, competition, and long-term promotion details.

The `Store` column is used to connect the daily records with the store-level metadata.

---

## What Was Done in This Notebook

The notebook follows a step-by-step workflow for understanding and preparing the Rossmann dataset.

### 1. Problem Understanding
We first introduced the forecasting problem and explained why predicting sales is useful in a retail setting. The goal was clearly defined as predicting the `Sales` variable from historical and store-related features.

### 2. Dataset and Feature Understanding
We reviewed the available files and described the variables found in `train.csv`, `test.csv`, and `store.csv`. This helped clarify the meaning of each feature before starting the analysis.

### 3. Importing Libraries and Loading Data
The required Python libraries were imported, and the dataset files were loaded into pandas DataFrames. File existence checks were included to ensure the notebook runs correctly only when all required files are available.

### 4. Exploratory Data Analysis (EDA)
Several visual analyses were performed to better understand the patterns in sales:

- **Average sales by day of week**  
  Sales were compared across the seven days of the week to identify weekly patterns.

- **Average sales with and without promotion**  
  A comparison was made between promotional and non-promotional days.

- **Average sales by month**  
  Monthly average sales were analyzed to observe seasonal patterns over the year.

- **Average sales by state holiday**  
  Sales were compared across different state holiday categories.

- **Average sales during school holidays vs non-school holidays**  
  This helped examine whether school holiday periods are associated with changes in sales.

These plots provided an initial understanding of how sales vary with time, promotions, and holiday-related factors.

## Key Insights from the Analysis

From the exploratory analysis, several patterns were observed:

- Sales vary across the days of the week.
- Promotions are associated with higher average sales.
- Monthly sales patterns suggest seasonality, especially toward the end of the year.
- State holidays appear to be linked with differences in average sales.
- School holidays show a smaller but noticeable effect on sales.

These observations help explain the structure of the dataset and provide useful context before moving to preprocessing and model building.


## Additional Data Preparation and Feature Engineering Steps

After the initial exploratory analysis, several preprocessing steps were added to improve the dataset and prepare it for machine learning.

### 5. Merging Store Metadata
The sales dataset was merged with `store.csv` using the `Store` column. This step added useful store-level information such as `StoreType`, `Assortment`, competition-related variables, and promotion-related variables to the main dataset.

### 6. Additional Analysis After Merging
Once the store information was added, more comparisons were made to understand how store characteristics affect sales:

- **Average sales by store type**  
  Store type **b** showed the highest average sales, while store types **a**, **c**, and **d** had fairly similar performance.

- **Average sales by assortment**  
  Assortment type **b** had the highest average sales, followed by **c**, while **a** had the lowest. This suggests that product assortment may influence store performance.

These comparisons showed that store-level attributes play an important role in explaining differences in sales.

### 7. Missing Value Inspection
Missing values were checked across all columns and sorted from highest to lowest. This helped identify which variables required cleaning before modeling.

### 8. Handling Missing Values
Special treatment was applied to missing values in promotion- and competition-related variables:

- Missing values in `PromoInterval` were replaced with `'None'`.
- Missing values in `Promo2SinceYear` and `Promo2SinceWeek` were filled with `0`.
- Missing values in `CompetitionOpenSinceYear` and `CompetitionOpenSinceMonth` were filled with `0`.
- Missing values in `CompetitionDistance` were replaced with the median value.

In addition, missing indicator columns were created for important variables to preserve information about whether a value was originally missing.

### 9. Date Feature Engineering
The `Date` column was converted into datetime format, and several time-based features were extracted:

- `Day`
- `Month`
- `Year`

Two additional binary features were also created:

- `IsStateHoliday` to indicate whether a record corresponds to a state holiday
- `IsWeekend` to indicate whether the day falls on a weekend

These engineered features make time-related patterns easier for the model to learn.

### 10. Removing the `Customers` Column
The `Customers` column was removed before modeling because it is strongly related to sales and may introduce data leakage. In a real forecasting setting, customer counts may not be known in advance, so excluding this variable makes the model evaluation more realistic.

### 11. Encoding Categorical Variables
Categorical variables such as `StateHoliday`, `StoreType`, `Assortment`, and `PromoInterval` were transformed using one-hot encoding. A `ColumnTransformer` was used so that only the selected categorical columns were encoded while the remaining variables were passed through unchanged.

### 12. Time-Based Data Splitting
The dataset was split into training and validation sets based on time:

- records before **2015-01-01** were used for training
- records from **2015-01-01** onward were used for validation

This approach preserves the chronological order of the data and creates a more realistic forecasting setup compared with random splitting.

### 13. Time Series Cross-Validation and Evaluation Metrics
`TimeSeriesSplit` was introduced to evaluate models in a way that respects the sequence of time-based data. An evaluation function was also defined to measure model performance using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **WAPE** (Weighted Absolute Percentage Error)

These metrics provide a broader view of model performance and help compare forecasting models more effectively.

## Updated Summary

In addition to the exploratory analysis, the notebook now includes important preprocessing and feature engineering steps needed for forecasting. The sales data was enriched with store metadata, missing values were handled carefully, time-based features were created, categorical variables were encoded, and the dataset was split using a time-based strategy. Together, these steps make the dataset more suitable for machine learning and create a stronger foundation for building reliable sales forecasting models.
