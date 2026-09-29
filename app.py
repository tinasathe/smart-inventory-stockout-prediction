import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Inventory AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #1f4e79;
    }

    .sub-title {
        font-size: 20px;
        color: #555;
    }

    .risk-high {
        background-color: #ffdddd;
        padding: 15px;
        border-radius: 10px;
        color: #b30000;
        font-weight: bold;
        text-align: center;
    }

    .risk-low {
        background-color: #ddffdd;
        padding: 15px;
        border-radius: 10px;
        color: #087f23;
        font-weight: bold;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA PATH
# =========================================================

DATA_PATH = "data/sales_data.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Date conversion
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Inventory ratio
    df["Inventory_Ratio"] = (
        df["Inventory Level"] /
        (df["Units Sold"] + 1)
    )

    # -----------------------------------------------------
    # Stockout / Risk target
    # -----------------------------------------------------

    df["Stockout"] = (
        df["Inventory Level"] <
        (1.5 * df["Units Sold"])
    ).astype(int)

    return df


# =========================================================
# TRAIN MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def train_models(df):

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    classification_features = [
        "Inventory Level",
        "Price",
        "Discount",
        "Promotion",
        "Competitor Pricing",
        "Epidemic",
        "Year",
        "Month",
        "DayOfWeek",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    X_classification = df[
        classification_features
    ].copy()

    y_classification = df["Stockout"]

    categorical_features = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numerical_features = [
        column
        for column in classification_features
        if column not in categorical_features
    ]

    classification_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    classifier = Pipeline(
        steps=[
            (
                "preprocessor",
                classification_preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    X_train_c, X_test_c, y_train_c, y_test_c = (
        train_test_split(
            X_classification,
            y_classification,
            test_size=0.20,
            random_state=42,
            stratify=y_classification
        )
    )

    classifier.fit(
        X_train_c,
        y_train_c
    )

    y_pred_c = classifier.predict(
        X_test_c
    )

    classification_metrics = {
        "Accuracy": accuracy_score(
            y_test_c,
            y_pred_c
        ),

        "Precision": precision_score(
            y_test_c,
            y_pred_c,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test_c,
            y_pred_c,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test_c,
            y_pred_c,
            zero_division=0
        )
    }


    # =====================================================
    # DEMAND REGRESSION
    # =====================================================

    regression_features = [
        "Inventory Level",
        "Price",
        "Discount",
        "Promotion",
        "Competitor Pricing",
        "Epidemic",
        "Year",
        "Month",
        "DayOfWeek",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    X_regression = df[
        regression_features
    ].copy()

    y_regression = df["Demand"]

    regression_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    regressor = Pipeline(
        steps=[
            (
                "preprocessor",
                regression_preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    X_train_r, X_test_r, y_train_r, y_test_r = (
        train_test_split(
            X_regression,
            y_regression,
            test_size=0.20,
            random_state=42
        )
    )

    regressor.fit(
        X_train_r,
        y_train_r
    )

    y_pred_r = regressor.predict(
        X_test_r
    )

    regression_metrics = {

        "MAE": mean_absolute_error(
            y_test_r,
            y_pred_r
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_test_r,
                y_pred_r
            )
        ),

        "R2": r2_score(
            y_test_r,
            y_pred_r
        )
    }

    return (
        classifier,
        regressor,
        classification_features,
        regression_features,
        classification_metrics,
        regression_metrics
    )


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = load_data()

except Exception as error:

    st.error(
        "❌ Dataset load झाला नाही."
    )

    st.code(
        f"""
Expected file:

{DATA_PATH}

Error:

{error}
        """
    )

    st.stop()


# =========================================================
# CHECK STOCKOUT TARGET
# =========================================================

if df["Stockout"].nunique() < 2:

    st.error(
        "Stockout target मध्ये दोन classes तयार झाले नाहीत."
    )

    st.info(
        "Inventory Level आणि Units Sold च्या values check करा."
    )

    st.stop()


# =========================================================
# TRAIN MODELS
# =========================================================

try:

    (
        classifier,
        regressor,
        classification_features,
        regression_features,
        classification_metrics,
        regression_metrics
    ) = train_models(df)

except Exception as error:

    st.error(
        "❌ Machine Learning models train झाले नाहीत."
    )

    st.exception(error)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📦 Smart Inventory AI")

st.sidebar.write(
    "Stockout & Restock Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔮 Prediction",
        "📈 Analytics",
        "🤖 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-powered inventory decision support system"
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '📦 Smart Inventory Stockout & Restock Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'AI-based Inventory Management System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.write(
        """
        This project uses Machine Learning to analyze retail inventory,
        predict product demand, identify stockout risk and recommend
        inventory replenishment quantities.
        """
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "🏪 Stores",
        f"{df['Store ID'].nunique():,}"
    )

    col3.metric(
        "📦 Products",
        f"{df['Product ID'].nunique():,}"
    )

    col4.metric(
        "🗂 Categories",
        f"{df['Category'].nunique():,}"
    )

    st.divider()

    # -----------------------------------------------------
    # OBJECTIVES
    # -----------------------------------------------------

    st.subheader(
        "🎯 Project Objectives"
    )

    objective_col1, objective_col2 = st.columns(2)

    with objective_col1:

        st.markdown(
            """
            ### 📈 Demand Prediction

            Predict expected product demand using:

            - Price
            - Discount
            - Promotion
            - Seasonality
            - Weather
            - Region
            - Competitor pricing
            """
        )

    with objective_col2:

        st.markdown(
            """
            ### 📦 Inventory Management

            System provides:

            - Stockout risk
            - Demand forecast
            - Recommended restock quantity
            - Inventory analytics
            """
        )

    st.divider()

    st.info(
        "Stockout risk target is created from historical inventory "
        "coverage using Inventory Level < 1.5 × Units Sold."
    )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Inventory Dashboard"
    )

    # -----------------------------------------------------
    # KPI
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Units Sold",
        f"{df['Units Sold'].sum():,.0f}"
    )

    col2.metric(
        "Average Inventory",
        f"{df['Inventory Level'].mean():,.1f}"
    )

    col3.metric(
        "Average Demand",
        f"{df['Demand'].mean():,.1f}"
    )

    col4.metric(
        "Stockout Risk Records",
        f"{df['Stockout'].sum():,}"
    )

    st.divider()

    # -----------------------------------------------------
    # DAILY SALES
    # -----------------------------------------------------

    st.subheader(
        "📈 Daily Sales Trend"
    )

    daily_sales = (
        df.groupby("Date")["Units Sold"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        daily_sales.index,
        daily_sales.values,
        color="#1f77b4"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Units Sold"
    )

    ax.set_title(
        "Daily Sales Trend"
    )

    ax.grid(
        alpha=0.2
    )

    fig.autofmt_xdate()

    st.pyplot(
        fig,
        clear_figure=True
    )

    # -----------------------------------------------------
    # CATEGORY SALES
    # -----------------------------------------------------

    st.subheader(
        "🗂 Category-wise Sales"
    )

    category_sales = (
        df.groupby("Category")["Units Sold"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_sales
    )

    # -----------------------------------------------------
    # REGION SALES
    # -----------------------------------------------------

    st.subheader(
        "🌍 Region-wise Sales"
    )

    region_sales = (
        df.groupby("Region")["Units Sold"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        region_sales
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Product Demand & Stockout Prediction"
    )

    st.write(
        "Enter product and inventory conditions to generate an AI prediction."
    )

    st.divider()

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        category = st.selectbox(
            "Category",
            sorted(
                df["Category"]
                .dropna()
                .unique()
            )
        )

        region = st.selectbox(
            "Region",
            sorted(
                df["Region"]
                .dropna()
                .unique()
            )
        )

        weather = st.selectbox(
            "Weather Condition",
            sorted(
                df["Weather Condition"]
                .dropna()
                .unique()
            )
        )

    with col2:

        seasonality = st.selectbox(
            "Seasonality",
            sorted(
                df["Seasonality"]
                .dropna()
                .unique()
            )
        )

        inventory = st.number_input(
            "Current Inventory",
            min_value=0.0,
            value=float(
                df["Inventory Level"].median()
            ),
            step=1.0
        )

        price = st.number_input(
            "Price",
            min_value=0.0,
            value=float(
                df["Price"].median()
            ),
            step=0.01
        )

    with col3:

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            value=float(
                df["Discount"].median()
            ),
            step=1.0
        )

        promotion = st.selectbox(
            "Promotion",
            sorted(
                df["Promotion"]
                .dropna()
                .unique()
            )
        )

        competitor_price = st.number_input(
            "Competitor Pricing",
            min_value=0.0,
            value=float(
                df["Competitor Pricing"].median()
            ),
            step=0.01
        )

    epidemic = st.selectbox(
        "Epidemic",
        sorted(
            df["Epidemic"]
            .dropna()
            .unique()
        )
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        lead_time = st.number_input(
            "Supplier Lead Time (days)",
            min_value=1,
            max_value=60,
            value=7,
            step=1
        )

    with col2:

        safety_stock = st.number_input(
            "Safety Stock",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚀 Generate Prediction",
        use_container_width=True
    ):

        current_date = pd.Timestamp.now()

        input_data = pd.DataFrame(
            [{
                "Inventory Level": inventory,
                "Price": price,
                "Discount": discount,
                "Promotion": promotion,
                "Competitor Pricing": competitor_price,
                "Epidemic": epidemic,
                "Year": current_date.year,
                "Month": current_date.month,
                "DayOfWeek": current_date.dayofweek,
                "Category": category,
                "Region": region,
                "Weather Condition": weather,
                "Seasonality": seasonality
            }]
        )

        # -------------------------------------------------
        # DEMAND PREDICTION
        # -------------------------------------------------

        predicted_demand = float(
            regressor.predict(
                input_data[
                    regression_features
                ]
            )[0]
        )

        predicted_demand = max(
            0,
            predicted_demand
        )

        # -------------------------------------------------
        # STOCKOUT PREDICTION
        # -------------------------------------------------

        stockout_class = int(
            classifier.predict(
                input_data[
                    classification_features
                ]
            )[0]
        )

        # Probability
        probability = float(
            classifier.predict_proba(
                input_data[
                    classification_features
                ]
            )[0][1]
        )

        # -------------------------------------------------
        # RESTOCK CALCULATION
        # -------------------------------------------------

        target_stock = (
            predicted_demand
            * lead_time
            + safety_stock
        )

        recommended_restock = max(
            0,
            target_stock - inventory
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.subheader(
            "📊 Prediction Results"
        )

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "📈 Predicted Demand",
            f"{predicted_demand:.0f} units"
        )

        result2.metric(
            "⚠️ Stockout Probability",
            f"{probability * 100:.1f}%"
        )

        result3.metric(
            "📦 Recommended Restock",
            f"{recommended_restock:.0f} units"
        )

        st.divider()

        # -------------------------------------------------
        # RISK DISPLAY
        # -------------------------------------------------

        if stockout_class == 1:

            st.markdown(
                """
                <div class="risk-high">
                ⚠️ HIGH STOCKOUT RISK
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="risk-low">
                ✅ LOW STOCKOUT RISK
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )

        st.write(
            f"Estimated stockout probability: "
            f"**{probability * 100:.2f}%**"
        )

        st.info(
            f"""
            Recommended Target Stock:

            Predicted Demand × Lead Time + Safety Stock

            = {predicted_demand:.1f}
            × {lead_time}
            + {safety_stock:.1f}

            = {target_stock:.1f} units
            """
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📈 Analytics":

    st.title(
        "📈 Inventory Analytics"
    )

    # -----------------------------------------------------
    # INVENTORY VS SALES
    # -----------------------------------------------------

    st.subheader(
        "📦 Inventory Level vs Units Sold"
    )

    sample_size = min(
        3000,
        len(df)
    )

    sample_df = df.sample(
        sample_size,
        random_state=42
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        sample_df["Inventory Level"],
        sample_df["Units Sold"],
        alpha=0.35,
        color="#ff7f0e"
    )

    ax.set_xlabel(
        "Inventory Level"
    )

    ax.set_ylabel(
        "Units Sold"
    )

    ax.set_title(
        "Inventory Level vs Units Sold"
    )

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    # -----------------------------------------------------
    # MONTHLY SALES
    # -----------------------------------------------------

    st.subheader(
        "📅 Monthly Sales Trend"
    )

    monthly_sales = (
        df.groupby(
            ["Year", "Month"]
        )["Units Sold"]
        .sum()
        .reset_index()
    )

    monthly_sales["Period"] = (
        monthly_sales["Year"]
        .astype(str)
        + "-"
        + monthly_sales["Month"]
        .astype(str)
        .str.zfill(2)
    )

    monthly_chart = (
        monthly_sales
        .set_index("Period")["Units Sold"]
    )

    st.line_chart(
        monthly_chart
    )

    # -----------------------------------------------------
    # REGION
    # -----------------------------------------------------

    st.subheader(
        "🌍 Region-wise Demand"
    )

    region_demand = (
        df.groupby("Region")["Demand"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        region_demand
    )

    # -----------------------------------------------------
    # SEASONALITY
    # -----------------------------------------------------

    st.subheader(
        "🌦 Seasonality-wise Demand"
    )

    season_demand = (
        df.groupby("Seasonality")["Demand"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        season_demand
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title(
        "🤖 Machine Learning Model Performance"
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    st.subheader(
        "⚠️ Stockout Classification"
    )

    st.caption(
        "Algorithm: Random Forest Classifier"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{classification_metrics['Accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{classification_metrics['Precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{classification_metrics['Recall'] * 100:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{classification_metrics['F1 Score'] * 100:.2f}%"
    )

    st.divider()

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    st.subheader(
        "📈 Demand Prediction"
    )

    st.caption(
        "Algorithm: Random Forest Regressor"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{regression_metrics['MAE']:.2f}"
    )

    c2.metric(
        "RMSE",
        f"{regression_metrics['RMSE']:.2f}"
    )

    c3.metric(
        "R² Score",
        f"{regression_metrics['R2']:.3f}"
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "📊 Dataset Information"
    )

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Rows",
        f"{len(df):,}"
    )

    info2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    info3.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    st.success(
        "Models are automatically trained when the application starts."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Smart Inventory Stockout & Restock Prediction"
)

st.sidebar.caption(
    "Machine Learning • Streamlit • Python"
)