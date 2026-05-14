# Rossmann Store Sales Forecasting

## Project Overview

This project focuses on forecasting daily sales for Rossmann stores using historical sales data and store-level information.

The goal is to predict the `Sales` value for each store on a given day. This is a supervised regression problem, where the model learns from historical sales records and store features to estimate future sales.

Accurate sales forecasting is important in retail because it can support better decisions in inventory planning, staffing, promotion planning, store operations, and financial management.

This notebook follows a complete machine learning workflow, including exploratory data analysis, preprocessing, feature engineering, leakage-safe validation, model training, model comparison, error analysis, feature importance, and final test prediction generation.

---

## Dataset

The project uses the Rossmann Store Sales dataset.

The main files used are:

- `train.csv`: historical daily sales data used for training and validation
- `test.csv`: unseen records used for final sales predictions
- `store.csv`: store-level information such as store type, assortment, competition distance, and promotion details

The `Store` column is used to merge the daily sales data with the store-level metadata.

---

## Problem Type

This is a supervised machine learning regression problem.

The target variable is:

- `Sales`

The aim is to build a model that can predict future store sales as accurately as possible.

---

## Notebook Workflow

## 1. Data Loading

The required datasets are loaded using pandas.

The training data is merged with the store metadata using the `Store` column. This gives each daily sales record additional store-level information, such as store type, assortment, competition details, and promotion-related variables.

---

## 2. Exploratory Data Analysis

Exploratory data analysis was performed to understand the main patterns in the dataset before building models.

The notebook analyzes average sales by:

- day of the week
- promotion status
- month
- state holiday
- school holiday
- store type
- assortment type

These visualizations help identify how sales change across time, promotions, holidays, and store characteristics.

### Key EDA Insights

Several useful patterns were observed:

- Sales vary across different days of the week.
- Promotions are associated with higher average sales.
- Monthly sales patterns suggest seasonality.
- State holidays affect sales behavior.
- School holidays have a smaller but noticeable effect on sales.
- Store type and assortment type influence average sales.
- Store type `b` and assortment type `b` show higher average sales compared with other categories.

These insights helped guide the preprocessing and feature engineering steps.

---

## 3. Data Preprocessing

A preprocessing function was created to clean the data and generate useful features.

The preprocessing step includes:

- converting the `Date` column into datetime format
- extracting date-based features:
  - `Day`
  - `Month`
  - `Year`
  - `WeekOfYear`
  - `DayOfYear`
  - `IsWeekend`
  - `IsMonthStart`
  - `IsMonthEnd`
- cleaning holiday information
- creating an `IsStateHoliday` feature
- filling missing categorical values with clear labels such as `None` or `unknown`
- filling missing values in `Open`
- filling missing values in `CompetitionDistance`
- filling missing competition and promotion date columns with `0`

These steps make the dataset cleaner and more suitable for machine learning models.

---

## 4. Time-Based Train-Validation Split

Because this is a sales forecasting problem, the data was split based on time instead of randomly.

The most recent 12 weeks were used as the validation set, while all earlier records were used for training.

This creates a more realistic evaluation setup because the model is trained on past data and tested on future data.

---

## 5. Handling Data Leakage

Special care was taken to avoid data leakage.

The `Customers` column was removed before modelling because customer count is highly related to sales and may not be known before predicting future sales. Using it could make the model unrealistically strong.

Missing values such as `CompetitionDistance` were filled using values calculated from the training data only. This prevents information from the validation period from influencing preprocessing decisions.

---

## 6. Historical Mean Sales Features

Several historical mean sales features were created to help the model learn store-level sales behavior.

Examples include:

- average sales by store
- average sales by store and day of week
- average sales by store and month
- average sales by store and promotion status
- average sales by store and school holiday
- average sales by store and state holiday
- average sales by store, day of week, and promotion status
- average sales by store type
- average sales by store type and day of week
- average sales by assortment type
- average sales by promotion status
- average sales by day of week
- average sales by month

These features allow the model to use historical sales patterns under different business conditions.

---

## 7. Leakage-Safe Mean Feature Creation

Mean sales features are powerful, but they must be created carefully because they use the target variable `Sales`.

To avoid target leakage in the training data, the notebook uses a time-based out-of-fold approach.

For each time fold:

1. Earlier dates are used to calculate historical mean sales features.
2. These features are applied to the next time period.
3. The model never receives mean features calculated from the same row or from future dates.

This ensures that each training row only uses information that would have been available in the past.

For the validation set, mean sales features are calculated using the full training data only. This reflects a realistic forecasting setup, because historical training sales are known before predicting the future validation period.

---

## 8. Feature Preparation

The final feature set excludes:

- `Sales`, because it is the target variable
- `Date`, because useful date features were already extracted
- `Customers`, because it may cause data leakage

The selected features are split into:

- `X_train`
- `y_train`
- `X_val`
- `y_val`

Categorical variables are handled using one-hot encoding.

The encoded categorical columns are:

- `StateHoliday`
- `StoreType`
- `Assortment`
- `PromoInterval`

A `ColumnTransformer` is used to apply one-hot encoding to the categorical columns while keeping the remaining numerical columns unchanged.

---

## 9. Evaluation Metrics

The models were evaluated using several regression metrics:

- `MAE`: Mean Absolute Error
- `RMSE`: Root Mean Squared Error
- `R²`: Coefficient of Determination
- `WAPE`: Weighted Absolute Percentage Error
- `RMSPE`: Root Mean Squared Percentage Error

Using multiple metrics gives a better understanding of model performance from different perspectives.

---

## 10. Models Trained

Four models were trained and compared.

### Baseline Model

A simple baseline model was created using the average sales of each store from the training data.

This provides a reference point to check whether the machine learning models actually improve performance.

### Decision Tree Regressor

A Decision Tree model was trained as a simple tree-based model.

It learns rules by splitting the data based on feature values such as store, promotion status, weekday, holidays, and store type.

### Random Forest Regressor

A Random Forest model was trained to improve stability and reduce overfitting.

Random Forest builds multiple decision trees and averages their predictions.

### XGBoost Regressor

An XGBoost model was trained as the main advanced model.

XGBoost builds decision trees sequentially, where each new tree tries to correct the errors made by the previous trees. This often makes it highly effective for structured/tabular datasets.

---

## 11. Model Comparison

The models were compared on the validation set.

| Rank | Model | MAE | RMSE | R² | WAPE | RMSPE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | XGB | 547.617 | 842.569 | 0.9524 | 9.26% | 12.75% |
| 2 | Random Forest | 565.708 | 884.138 | 0.9475 | 9.57% | 13.80% |
| 3 | Decision Tree | 791.304 | 1229.064 | 0.8986 | 13.38% | 19.66% |
| 4 | Baseline | 2392.107 | 3265.907 | 0.2841 | 40.45% | 29.60% |

The XGBoost model achieved the best validation performance. It had the lowest MAE, RMSE, WAPE, and RMSPE, as well as the highest R² score.

This means XGBoost produced the most accurate predictions among the tested models.

---

## 12. Error Analysis

Additional analysis was performed for the best model, XGBoost.

### Actual vs Predicted Sales

A scatter plot was used to compare actual sales values with predicted sales values.

Points close to the diagonal line indicate accurate predictions, while points farther from the line represent larger prediction errors.

### Residual Plot

A residual plot was created to inspect the model errors.

Residuals were calculated as:

```text
Actual Sales - Predicted Sales

```

## Streamlit App

This project also includes a Streamlit app that allows users to predict sales for a selected Rossmann store and date.

The app loads the trained XGBoost pipeline, applies the same preprocessing and feature engineering steps used in the notebook, and returns the predicted sales value.

The app uses the full available training history to create historical mean sales features for new predictions. This reflects a realistic deployment setup because past sales data is already known when predicting future sales.

To run the app:

```bash
streamlit run app.py
```

---

## How to Run the Project

1. Clone this repository.
2. Place the dataset files inside the `Data` folder.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run the notebook cells from top to bottom.
5. Save the trained model to the `models` folder.
6. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Data Note

The dataset is not included in this repository. It can be downloaded from the Rossmann Store Sales competition page on Kaggle.

Expected folder structure:

```text
project-folder/
│
├── Data/
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
│
├── models/
│   └── rossmann_xgb_pipeline.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── Rossmann.ipynb
```

---

## Final Result

The best-performing model was XGBoost.

Validation performance:

- MAE: 547.617
- RMSE: 842.569
- R²: 0.9524
- WAPE: 9.26%
- RMSPE: 12.75%

These results show that the final model captures the main sales patterns well and performs much better than the baseline model.

---

## Conclusion

This project demonstrates a complete machine learning workflow for retail sales forecasting.

The notebook includes:

- exploratory data analysis
- missing value handling
- date-based feature engineering
- leakage-safe historical mean sales features
- time-based train-validation splitting
- categorical encoding
- baseline modelling
- Decision Tree, Random Forest, and XGBoost models
- model comparison using multiple metrics
- error analysis
- feature importance analysis
- final test prediction generation
- Streamlit app deployment

The final XGBoost model achieved the strongest validation performance and was selected for generating the final sales predictions.