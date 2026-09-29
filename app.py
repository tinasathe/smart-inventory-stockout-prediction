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

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1d4ed8
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 18px;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }

    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    possible_paths = [
        "data/sales_data.csv",
        "sales_data.csv"
    ]

    df = None

    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError(
            "sales_data.csv not found. "
            "Place it inside data/ folder or project root."
        )

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    # -----------------------------------------------------
    # SORT DATA
    # -----------------------------------------------------

    df = df.sort_values(
        ["Store ID", "Product ID", "Date"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # DATE FEATURES
    # -----------------------------------------------------

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # -----------------------------------------------------
    # FUTURE DEMAND
    # -----------------------------------------------------

    # Approximate next 7 observations' demand
    # for the same Store + Product.
    #
    # This is used only to create the historical
    # stockout-risk target.

    grouped = df.groupby(
        ["Store ID", "Product ID"]
    )["Units Sold"]

    future_columns = []

    for i in range(1, 8):

        shifted = grouped.shift(-i)

        future_columns.append(
            shifted.fillna(0)
        )

    df["Future_7_Day_Demand"] = sum(
        future_columns
    )

    # -----------------------------------------------------
    # STOCKOUT TARGET
    # -----------------------------------------------------

    df["Stockout"] = (
        df["Inventory Level"]
        < df["Future_7_Day_Demand"]
    ).astype(int)

    # -----------------------------------------------------
    # REMOVE LAST RECORDS WITHOUT FUTURE DEMAND
    # -----------------------------------------------------

    # Last observations of each product/store do not have
    # a complete future 7-day window.
    valid_future = df["Future_7_Day_Demand"] > 0

    model_df = df[
        valid_future
    ].copy()

    # Fallback in case dataset structure creates no
    # complete future records.
    if len(model_df) < 100:

        model_df = df.copy()

        model_df["Stockout"] = (
            df["Inventory Level"]
            < (df["Units Sold"] * 1.5)
        ).astype(int)

    return df, model_df


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_models(model_df):

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
        "WeekOfYear",

        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    X_classification = model_df[
        classification_features
    ]

    y_classification = model_df[
        "Stockout"
    ]

    categorical_classification = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numerical_classification = [
        c for c in classification_features
        if c not in categorical_classification
    ]

    classification_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_classification
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_classification
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

    # Check if target has two classes

    if y_classification.nunique() < 2:

        raise ValueError(
            "Stockout target contains only one class. "
            "The dataset does not contain enough variation "
            "for classification."
        )

    Xc_train, Xc_test, yc_train, yc_test = (
        train_test_split(
            X_classification,
            y_classification,
            test_size=0.20,
            random_state=42,
            stratify=y_classification
        )
    )

    classifier.fit(
        Xc_train,
        yc_train
    )

    yc_pred = classifier.predict(
        Xc_test
    )

    classification_metrics = {

        "Accuracy": accuracy_score(
            yc_test,
            yc_pred
        ),

        "Precision": precision_score(
            yc_test,
            yc_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            yc_test,
            yc_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            yc_test,
            yc_pred,
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
        "WeekOfYear",

        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    X_regression = model_df[
        regression_features
    ]

    y_regression = model_df[
        "Demand"
    ]

    categorical_regression = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numerical_regression = [
        c for c in regression_features
        if c not in categorical_regression
    ]

    regression_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_regression
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_regression
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

    Xr_train, Xr_test, yr_train, yr_test = (
        train_test_split(
            X_regression,
            y_regression,
            test_size=0.20,
            random_state=42
        )
    )

    regressor.fit(
        Xr_train,
        yr_train
    )

    yr_pred = regressor.predict(
        Xr_test
    )

    regression_metrics = {

        "MAE": mean_absolute_error(
            yr_test,
            yr_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                yr_test,
                yr_pred
            )
        ),

        "R2": r2_score(
            yr_test,
            yr_pred
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

    df, model_df = load_data()

except Exception as e:

    st.error(
        "❌ Dataset load झाला नाही."
    )

    st.code(
        str(e)
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

    ) = train_models(
        model_df
    )

except Exception as e:

    st.error(
        "❌ Model training failed."
    )

    st.exception(e)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📦 Smart Inventory AI"
)

st.sidebar.caption(
    "Stockout • Demand • Restock Prediction"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Home",
        "📊 Dashboard",
        "🔮 Prediction",
        "📈 Analytics",
        "🚨 Stockout Alerts",
        "🤖 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-powered inventory decision support system"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>📦 Smart Inventory AI</h1>

        <p>
        Stockout Prediction • Demand Forecasting •
        Restock Recommendation
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "🎯 Project Overview"
    )

    st.write(
        """
        Smart Inventory AI is a Machine Learning based inventory
        decision-support system.

        The system analyzes historical retail data and provides:

        • Product demand prediction  
        • Stockout risk prediction  
        • Inventory health monitoring  
        • Restock quantity recommendation  
        • Store and product analytics
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📊 Total Records",
        f"{len(df):,}"
    )

    c2.metric(
        "📦 Products",
        f"{df['Product ID'].nunique():,}"
    )

    c3.metric(
        "🏪 Stores",
        f"{df['Store ID'].nunique():,}"
    )

    c4.metric(
        "🏷 Categories",
        f"{df['Category'].nunique():,}"
    )

    st.divider()

    st.subheader(
        "🤖 Machine Learning Algorithms"
    )

    a1, a2 = st.columns(2)

    with a1:

        st.info(
            """
            **Random Forest Classifier**

            Used for stockout-risk classification.

            Output:
            LOW / HIGH stockout risk
            """
        )

    with a2:

        st.info(
            """
            **Random Forest Regressor**

            Used for demand prediction.

            Output:
            Predicted demand in units
            """
        )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Inventory Dashboard"
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    f1, f2, f3 = st.columns(3)

    with f1:

        selected_region = st.selectbox(
            "🌍 Region",
            ["All"] +
            sorted(
                df["Region"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with f2:

        selected_category = st.selectbox(
            "🏷 Category",
            ["All"] +
            sorted(
                df["Category"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with f3:

        selected_store = st.selectbox(
            "🏪 Store",
            ["All"] +
            sorted(
                df["Store ID"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    dashboard_df = df.copy()

    if selected_region != "All":

        dashboard_df = dashboard_df[
            dashboard_df["Region"]
            == selected_region
        ]

    if selected_category != "All":

        dashboard_df = dashboard_df[
            dashboard_df["Category"]
            == selected_category
        ]

    if selected_store != "All":

        dashboard_df = dashboard_df[
            dashboard_df["Store ID"]
            == selected_store
        ]

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    total_sales = dashboard_df[
        "Units Sold"
    ].sum()

    avg_inventory = dashboard_df[
        "Inventory Level"
    ].mean()

    avg_demand = dashboard_df[
        "Demand"
    ].mean()

    low_stock_count = (
        dashboard_df["Inventory Level"]
        < dashboard_df["Units Sold"]
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📈 Total Units Sold",
        f"{total_sales:,.0f}"
    )

    c2.metric(
        "📦 Avg Inventory",
        f"{avg_inventory:,.1f}"
    )

    c3.metric(
        "🎯 Avg Demand",
        f"{avg_demand:,.1f}"
    )

    c4.metric(
        "🚨 Low Stock Records",
        f"{low_stock_count:,}"
    )

    st.divider()

    # -----------------------------------------------------
    # DAILY SALES
    # -----------------------------------------------------

    st.subheader(
        "📈 Daily Sales Trend"
    )

    daily_sales = (
        dashboard_df
        .groupby("Date")["Units Sold"]
        .sum()
    )

    st.line_chart(
        daily_sales
    )

    # -----------------------------------------------------
    # CATEGORY SALES
    # -----------------------------------------------------

    st.subheader(
        "🏷 Category-wise Sales"
    )

    category_sales = (
        dashboard_df
        .groupby("Category")["Units Sold"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_sales
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.download_button(

        "📥 Download Filtered Inventory Data",

        dashboard_df.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="filtered_inventory.csv",

        mime="text/csv"
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Product Demand & Stockout Prediction"
    )

    st.write(
        "Select a product and store to generate an AI prediction."
    )

    st.divider()

    # -----------------------------------------------------
    # PRODUCT + STORE
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        selected_product = st.selectbox(
            "📦 Select Product",
            sorted(
                df["Product ID"]
                .dropna()
                .unique()
            )
        )

    with c2:

        selected_store = st.selectbox(
            "🏪 Select Store",
            sorted(
                df["Store ID"]
                .dropna()
                .unique()
            )
        )

    selected_data = df[
        (
            df["Product ID"]
            == selected_product
        )
        &
        (
            df["Store ID"]
            == selected_store
        )
    ]

    if len(selected_data) == 0:

        st.warning(
            "No data available for this Product + Store combination."
        )

    else:

        product_data = (
            selected_data
            .sort_values("Date")
            .iloc[-1]
        )

        # -------------------------------------------------
        # CURRENT INFORMATION
        # -------------------------------------------------

        st.subheader(
            "📋 Current Inventory Information"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "📦 Current Inventory",
            f"{product_data['Inventory Level']:.0f}"
        )

        c2.metric(
            "📈 Last Units Sold",
            f"{product_data['Units Sold']:.0f}"
        )

        c3.metric(
            "💰 Price",
            f"₹{product_data['Price']:.2f}"
        )

        c4.metric(
            "🏷 Category",
            str(product_data["Category"])
        )

        st.divider()

        # -------------------------------------------------
        # SETTINGS
        # -------------------------------------------------

        st.subheader(
            "⚙️ Prediction Settings"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            lead_time = st.number_input(
                "🚚 Supplier Lead Time (days)",
                min_value=1,
                max_value=60,
                value=7
            )

        with c2:

            safety_stock = st.number_input(
                "🛡 Safety Stock",
                min_value=0.0,
                value=20.0,
                step=1.0
            )

        with c3:

            planning_days = st.number_input(
                "📅 Planning Horizon (days)",
                min_value=1,
                max_value=30,
                value=7
            )

        st.divider()

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button(
            "🚀 Generate AI Prediction",
            use_container_width=True
        ):

            input_data = pd.DataFrame(
                [{
                    "Inventory Level":
                        product_data[
                            "Inventory Level"
                        ],

                    "Price":
                        product_data["Price"],

                    "Discount":
                        product_data["Discount"],

                    "Promotion":
                        product_data["Promotion"],

                    "Competitor Pricing":
                        product_data[
                            "Competitor Pricing"
                        ],

                    "Epidemic":
                        product_data["Epidemic"],

                    "Year":
                        product_data["Year"],

                    "Month":
                        product_data["Month"],

                    "DayOfWeek":
                        product_data[
                            "DayOfWeek"
                        ],

                    "WeekOfYear":
                        product_data[
                            "WeekOfYear"
                        ],

                    "Category":
                        product_data["Category"],

                    "Region":
                        product_data["Region"],

                    "Weather Condition":
                        product_data[
                            "Weather Condition"
                        ],

                    "Seasonality":
                        product_data[
                            "Seasonality"
                        ]
                }]
            )

            # ---------------------------------------------
            # DEMAND
            # ---------------------------------------------

            predicted_daily_demand = float(
                regressor.predict(
                    input_data[
                        regression_features
                    ]
                )[0]
            )

            predicted_daily_demand = max(
                0,
                predicted_daily_demand
            )

            # ---------------------------------------------
            # STOCKOUT
            # ---------------------------------------------

            stockout_class = int(
                classifier.predict(
                    input_data[
                        classification_features
                    ]
                )[0]
            )

            probability = float(
                classifier.predict_proba(
                    input_data[
                        classification_features
                    ]
                )[0][1]
            )

            # ---------------------------------------------
            # RESTOCK
            # ---------------------------------------------

            current_inventory = float(
                product_data[
                    "Inventory Level"
                ]
            )

            forecast_demand = (
                predicted_daily_demand
                * planning_days
            )

            lead_time_demand = (
                predicted_daily_demand
                * lead_time
            )

            target_stock = (
                lead_time_demand
                + safety_stock
            )

            recommended_restock = max(
                0,
                target_stock
                - current_inventory
            )

            # ---------------------------------------------
            # RESULTS
            # ---------------------------------------------

            st.success(
                "✅ AI prediction generated successfully!"
            )

            st.subheader(
                "📊 Prediction Results"
            )

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "📈 Daily Demand",
                f"{predicted_daily_demand:.0f}"
            )

            r2.metric(
                f"📅 {planning_days}-Day Demand",
                f"{forecast_demand:.0f}"
            )

            r3.metric(
                "🚨 Stockout Probability",
                f"{probability * 100:.1f}%"
            )

            r4.metric(
                "📦 Recommended Restock",
                f"{recommended_restock:.0f}"
            )

            st.divider()

            # ---------------------------------------------
            # RISK
            # ---------------------------------------------

            if stockout_class == 1:

                st.markdown(
                    """
                    <div class="risk-high">
                    🚨 HIGH STOCKOUT RISK
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

            # ---------------------------------------------
            # RESTOCK EXPLANATION
            # ---------------------------------------------

            st.subheader(
                "📦 Restock Recommendation"
            )

            st.info(
                f"""
**Current Inventory:** {current_inventory:.0f} units

**Predicted Daily Demand:** {predicted_daily_demand:.0f} units

**Supplier Lead Time:** {lead_time} days

**Lead Time Demand:** {lead_time_demand:.0f} units

**Safety Stock:** {safety_stock:.0f} units

**Target Stock:** {target_stock:.0f} units

### Recommended Restock: {recommended_restock:.0f} units
"""
            )

            # ---------------------------------------------
            # PRODUCT HISTORY
            # ---------------------------------------------

            with st.expander(
                "🔎 View Product History"
            ):

                history = (
                    selected_data
                    .sort_values(
                        "Date",
                        ascending=False
                    )
                    .head(30)
                )

                st.dataframe(
                    history,
                    use_container_width=True
                )

            # ---------------------------------------------
            # DOWNLOAD PREDICTION
            # ---------------------------------------------

            result_df = pd.DataFrame(
                [{
                    "Product ID":
                        selected_product,

                    "Store ID":
                        selected_store,

                    "Current Inventory":
                        current_inventory,

                    "Predicted Daily Demand":
                        predicted_daily_demand,

                    "Planning Horizon":
                        planning_days,

                    "Predicted Horizon Demand":
                        forecast_demand,

                    "Stockout Probability":
                        probability,

                    "Stockout Risk":
                        "HIGH"
                        if stockout_class == 1
                        else "LOW",

                    "Lead Time":
                        lead_time,

                    "Safety Stock":
                        safety_stock,

                    "Recommended Restock":
                        recommended_restock
                }]
            )

            st.download_button(

                "📥 Download Prediction Report",

                result_df.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name=(
                    f"{selected_product}_"
                    f"{selected_store}_prediction.csv"
                ),

                mime="text/csv"
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
        color="#2563eb"
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

    st.line_chart(
        monthly_sales
        .set_index("Period")[
            "Units Sold"
        ]
    )

    # -----------------------------------------------------
    # REGION
    # -----------------------------------------------------

    st.subheader(
        "🌍 Region-wise Demand"
    )

    region_demand = (
        df.groupby(
            "Region"
        )["Demand"]
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
        df.groupby(
            "Seasonality"
        )["Demand"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        season_demand
    )

    # -----------------------------------------------------
    # CATEGORY DEMAND
    # -----------------------------------------------------

    st.subheader(
        "🏷 Category-wise Average Demand"
    )

    category_demand = (
        df.groupby(
            "Category"
        )["Demand"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_demand
    )


# =========================================================
# STOCKOUT ALERTS
# =========================================================

elif page == "🚨 Stockout Alerts":

    st.title(
        "🚨 Stockout Risk & Low Inventory Alerts"
    )

    st.write(
        """
        This page identifies products that may require
        inventory attention.
        """
    )

    alert_df = df.copy()

    # Current low-stock rule
    alert_df["Inventory_Coverage"] = (
        alert_df["Inventory Level"]
        / (alert_df["Units Sold"] + 1)
    )

    alert_df["Alert"] = np.where(
        alert_df["Inventory Level"]
        <= alert_df["Units Sold"],
        "HIGH",
        np.where(
            alert_df["Inventory_Coverage"] < 2,
            "MEDIUM",
            "LOW"
        )
    )

    high_alerts = alert_df[
        alert_df["Alert"] == "HIGH"
    ]

    medium_alerts = alert_df[
        alert_df["Alert"] == "MEDIUM"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🚨 HIGH",
        f"{len(high_alerts):,}"
    )

    c2.metric(
        "⚠️ MEDIUM",
        f"{len(medium_alerts):,}"
    )

    c3.metric(
        "🟢 LOW",
        f"{len(alert_df[alert_df['Alert'] == 'LOW']):,}"
    )

    st.divider()

    st.subheader(
        "🚨 Products Requiring Attention"
    )

    display_alerts = (
        alert_df[
            alert_df["Alert"].isin(
                ["HIGH", "MEDIUM"]
            )
        ]
        .sort_values(
            "Inventory_Coverage"
        )
        [
            [
                "Date",
                "Store ID",
                "Product ID",
                "Category",
                "Region",
                "Inventory Level",
                "Units Sold",
                "Demand",
                "Inventory_Coverage",
                "Alert"
            ]
        ]
        .head(100)
    )

    st.dataframe(
        display_alerts,
        use_container_width=True
    )

    st.download_button(

        "📥 Download Alert Report",

        display_alerts.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="stockout_alert_report.csv",

        mime="text/csv"
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
        "🚨 Stockout Classification"
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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    c3.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    c4.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )

    st.divider()

    # -----------------------------------------------------
    # TARGET DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "🚨 Stockout Target Distribution"
    )

    stockout_counts = (
        model_df["Stockout"]
        .value_counts()
        .rename(
            {
                0: "No Stockout Risk",
                1: "Stockout Risk"
            }
        )
    )

    st.bar_chart(
        stockout_counts
    )

    st.success(
        "Machine Learning models are trained automatically "
        "when the application starts."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Smart Inventory Stockout & Restock Prediction"
)

st.sidebar.caption(
    "Python • Pandas • Scikit-learn • Streamlit"
)