import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "models/rossmann_xgb_pipeline.pkl"
STORE_PATH = "Data/store.csv"
TRAIN_PATH = "Data/train.csv"





# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="🏪",
    layout="wide"
)

st.title("Rossmann Store Sales Forecast App")

st.write(
    "This app predicts daily sales for a selected Rossmann store using the trained XGBoost model."
)


# ============================================================
# File Checks
# ============================================================

def check_file(path_name, path_value):
    """
    Check whether a required file exists.
    Stop the app if the file is missing.
    """
    if not os.path.exists(path_value):
        st.error(f"{path_name} not found at: {path_value}")
        st.stop()

    if path_name == "MODEL_PATH" and os.path.getsize(path_value) == 0:
        st.error(f"Model file is empty: {path_value}")
        st.stop()


for name, path in {
    "MODEL_PATH": MODEL_PATH,
    "STORE_PATH": STORE_PATH,
    "TRAIN_PATH": TRAIN_PATH,
}.items():
    check_file(name, path)


# ============================================================
# Load Model and Data
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_store_data():
    return pd.read_csv(STORE_PATH)


@st.cache_data
def load_train_data():
    df = pd.read_csv(
        TRAIN_PATH,
        dtype={"StateHoliday": str},
        low_memory=False
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


model = load_model()
store_df = load_store_data()
raw_train_df = load_train_data()


# ============================================================
# Mean Feature Specifications
# ============================================================

MEAN_FEATURE_SPECS = [
    (["Store"], "MeanSales_ByStore"),
    (["Store", "DayOfWeek"], "MeanSales_ByStore_DayOfWeek"),
    (["Store", "Month"], "MeanSales_ByStore_Month"),
    (["Store", "Promo"], "MeanSales_ByStore_Promo"),
    (["Store", "SchoolHoliday"], "MeanSales_ByStore_SchoolHoliday"),
    (["Store", "StateHoliday"], "MeanSales_ByStore_StateHoliday"),
    (["Store", "DayOfWeek", "Promo"], "MeanSales_ByStore_DayOfWeek_Promo"),
    (["StoreType"], "MeanSales_ByStoreType"),
    (["StoreType", "DayOfWeek"], "MeanSales_ByStoreType_DayOfWeek"),
    (["Assortment"], "MeanSales_ByAssortment"),
    (["Promo"], "MeanSales_ByPromo"),
    (["DayOfWeek"], "MeanSales_ByDayOfWeek"),
    (["Month"], "MeanSales_ByMonth"),
]

EXPECTED_MEAN_COLS = [new_col for _, new_col in MEAN_FEATURE_SPECS]


# ============================================================
# Feature Engineering Functions
# ============================================================

def prepare_base_features(df):
    """
    Apply the same basic preprocessing and date feature engineering
    used in the notebook.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)
    else:
        df["StateHoliday"] = "0"

    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)

    if "PromoInterval" in df.columns:
        df["PromoInterval"] = df["PromoInterval"].fillna("None")
    else:
        df["PromoInterval"] = "None"

    if "StoreType" in df.columns:
        df["StoreType"] = df["StoreType"].fillna("unknown")
    else:
        df["StoreType"] = "unknown"

    if "Assortment" in df.columns:
        df["Assortment"] = df["Assortment"].fillna("unknown")
    else:
        df["Assortment"] = "unknown"

    if "Open" in df.columns:
        df["Open"] = df["Open"].fillna(1)
    else:
        df["Open"] = 1

    if "Promo" not in df.columns:
        df["Promo"] = 0

    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0

    if "CompetitionDistance" not in df.columns:
        df["CompetitionDistance"] = np.nan

    for col in [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    return df


def add_group_mean_from_source(source_df, target_df, group_cols, new_col):
    """
    Calculate grouped average sales from the source data
    and merge them into the target data.
    """
    source_df = source_df.copy()
    target_df = target_df.copy()

    missing_cols = [
        col for col in group_cols
        if col not in source_df.columns or col not in target_df.columns
    ]

    if missing_cols:
        target_df[new_col] = np.nan
        return target_df

    stats = (
        source_df
        .groupby(group_cols, dropna=False)["Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Sales": new_col})
    )

    target_df = target_df.merge(stats, on=group_cols, how="left")

    return target_df


def apply_backoff_fills(df_part, global_mean):
    """
    Fill missing mean sales features using a fallback strategy.
    More specific averages fall back to more general averages.
    """
    df_part = df_part.copy()

    for col in EXPECTED_MEAN_COLS:
        if col not in df_part.columns:
            df_part[col] = np.nan

    df_part["MeanSales_ByStore"] = df_part["MeanSales_ByStore"].fillna(global_mean)

    df_part["MeanSales_ByStore_DayOfWeek"] = df_part["MeanSales_ByStore_DayOfWeek"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStore_Month"] = df_part["MeanSales_ByStore_Month"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStore_Promo"] = df_part["MeanSales_ByStore_Promo"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStore_SchoolHoliday"] = df_part["MeanSales_ByStore_SchoolHoliday"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStore_StateHoliday"] = df_part["MeanSales_ByStore_StateHoliday"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStore_DayOfWeek_Promo"] = df_part[
        "MeanSales_ByStore_DayOfWeek_Promo"
    ].fillna(df_part["MeanSales_ByStore_DayOfWeek"])

    df_part["MeanSales_ByStoreType"] = df_part["MeanSales_ByStoreType"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByStoreType_DayOfWeek"] = df_part[
        "MeanSales_ByStoreType_DayOfWeek"
    ].fillna(df_part["MeanSales_ByStoreType"])

    df_part["MeanSales_ByAssortment"] = df_part["MeanSales_ByAssortment"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByPromo"] = df_part["MeanSales_ByPromo"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByDayOfWeek"] = df_part["MeanSales_ByDayOfWeek"].fillna(
        df_part["MeanSales_ByStore"]
    )

    df_part["MeanSales_ByMonth"] = df_part["MeanSales_ByMonth"].fillna(
        df_part["MeanSales_ByStore"]
    )

    return df_part


def add_train_only_mean_features(train_part, apply_part):
    """
    Create historical mean sales features using only the training/history data.
    """
    train_part = train_part.copy()
    apply_part = apply_part.copy()

    global_mean = train_part["Sales"].mean()

    for group_cols, new_col in MEAN_FEATURE_SPECS:
        apply_part = add_group_mean_from_source(
            source_df=train_part,
            target_df=apply_part,
            group_cols=group_cols,
            new_col=new_col
        )

    apply_part = apply_backoff_fills(apply_part, global_mean)

    return apply_part


# ============================================================
# Build Historical Reference Data
# ============================================================

@st.cache_data
def build_reference_data(raw_train_df, store_df):
    """
    Build the historical data used for creating mean sales features
    and filling missing values.

    For the deployed app, the full available training history is used.
    This matches the final prediction logic, where all historical sales
    are known before predicting future sales.
    """
    full_history = raw_train_df.merge(store_df, on="Store", how="left")
    full_history = prepare_base_features(full_history)

    competition_distance_fill = full_history["CompetitionDistance"].median()

    if pd.isna(competition_distance_fill):
        competition_distance_fill = 0.0

    full_history["CompetitionDistance"] = full_history["CompetitionDistance"].fillna(
        competition_distance_fill
    )

    return full_history, competition_distance_fill


full_history, competition_distance_fill = build_reference_data(
    raw_train_df,
    store_df
)


# ============================================================
# Get Expected Feature Columns from Saved Model
# ============================================================

def get_expected_columns(model):
    """
    Try to get the feature columns that were used when training the model.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


feature_cols = get_expected_columns(model)

if feature_cols is None:
    st.error(
        "Could not detect the model feature columns. "
        "Please make sure the saved model was trained with a pandas DataFrame."
    )
    st.stop()


# ============================================================
# Build Manual Input
# ============================================================

def build_manual_raw_input(
    store,
    date,
    promo,
    schoolholiday,
    stateholiday,
    open_status,
    store_df
):
    """
    Build one raw input row from the Streamlit UI.
    Store-level information is taken from store.csv.
    """
    store_row_df = store_df[store_df["Store"] == store]

    if store_row_df.empty:
        raise ValueError(f"Store {store} was not found in store.csv")

    store_row = store_row_df.iloc[0]
    pred_date = pd.to_datetime(date)

    raw_input = pd.DataFrame([{
        "Store": store,
        "DayOfWeek": pred_date.dayofweek + 1,
        "Date": pred_date,
        "Open": open_status,
        "Promo": promo,
        "StateHoliday": str(stateholiday),
        "SchoolHoliday": schoolholiday,
        "StoreType": store_row["StoreType"],
        "Assortment": store_row["Assortment"],
        "CompetitionDistance": store_row["CompetitionDistance"],
        "CompetitionOpenSinceMonth": store_row["CompetitionOpenSinceMonth"],
        "CompetitionOpenSinceYear": store_row["CompetitionOpenSinceYear"],
        "Promo2": store_row["Promo2"],
        "Promo2SinceWeek": store_row["Promo2SinceWeek"],
        "Promo2SinceYear": store_row["Promo2SinceYear"],
        "PromoInterval": store_row["PromoInterval"],
    }])

    return raw_input


def build_processed_input(
    raw_input,
    full_history,
    competition_distance_fill,
    feature_cols
):
    """
    Apply the same preprocessing and feature engineering steps
    before sending the input to the trained model.
    """
    input_df = prepare_base_features(raw_input)

    input_df["CompetitionDistance"] = input_df["CompetitionDistance"].fillna(
        competition_distance_fill
    )

    input_df = add_train_only_mean_features(
        train_part=full_history,
        apply_part=input_df
    )

    input_df = input_df.drop(
        columns=["Sales", "Date", "Customers"],
        errors="ignore"
    )

    missing_cols = [col for col in feature_cols if col not in input_df.columns]
    extra_cols = [col for col in input_df.columns if col not in feature_cols]

    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df.drop(columns=extra_cols, errors="ignore")
    input_df = input_df[feature_cols]

    return input_df, missing_cols, extra_cols


# ============================================================
# Streamlit User Interface
# ============================================================

st.subheader("Input Store Information")

col1, col2, col3 = st.columns(3)

with col1:
    store = st.number_input(
        "Store",
        min_value=int(store_df["Store"].min()),
        max_value=int(store_df["Store"].max()),
        step=1,
        value=1
    )

with col2:
    date = st.date_input(
        "Date",
        value=dt.date(2015, 8, 1),
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date(2035, 12, 31)
    )

with col3:
    open_status = st.selectbox(
        "Open",
        options=[1, 0],
        index=0,
        help="1 means the store is open, 0 means the store is closed."
    )

col4, col5, col6 = st.columns(3)

with col4:
    promo = st.selectbox(
        "Promo",
        options=[0, 1],
        index=0,
        help="1 means there is a promotion, 0 means there is no promotion."
    )

with col5:
    schoolholiday = st.selectbox(
        "SchoolHoliday",
        options=[0, 1],
        index=0
    )

with col6:
    stateholiday = st.selectbox(
        "StateHoliday",
        options=["0", "a", "b", "c"],
        index=0,
        help="0 means no state holiday."
    )


# ============================================================
# Prediction
# ============================================================

if st.button("Predict Sales", type="primary"):
    try:
        raw_input = build_manual_raw_input(
            store=store,
            date=date,
            promo=promo,
            schoolholiday=schoolholiday,
            stateholiday=stateholiday,
            open_status=open_status,
            store_df=store_df
        )

        input_df, missing_cols, extra_cols = build_processed_input(
            raw_input=raw_input,
            full_history=full_history,
            competition_distance_fill=competition_distance_fill,
            feature_cols=feature_cols
        )

        prediction = float(model.predict(input_df)[0])
        prediction = max(0, prediction)

        if open_status == 0:
            prediction = 0.0

        st.success(f"Predicted Sales: €{prediction:,.2f}")

        if missing_cols:
            st.warning(f"Missing columns were filled with 0: {missing_cols}")

        with st.expander("Show raw input"):
            st.dataframe(raw_input, use_container_width=True)

        with st.expander("Show processed input sent to model"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")