
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# SMART INVENTORY STOCKOUT & RESTOCK PREDICTION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Inventory AI",
    page_icon="📦",
    layout="wide"
)

DATA_PATH = "data/sales_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # A practical inventory-risk definition:
    # risk = current inventory is less than 1.5 times observed daily sales.
    # This is used only to create the historical classification target.
    df["Stockout"] = (
        df["Inventory Level"] < (1.5 * df["Units Sold"])
    ).astype(int)

    return df


@st.cache_resource
def train_models(df):
    # -------------------------
    # Classification model
    # -------------------------
    clf_features = [
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

    Xc = df[clf_features].copy()
    yc = df["Stockout"]

    categorical_clf = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numeric_clf = [c for c in clf_features if c not in categorical_clf]

    preprocessor_clf = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_clf),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_clf)
        ]
    )

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor_clf),
            ("model", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc,
        test_size=0.20,
        random_state=42,
        stratify=yc
    )

    classifier.fit(Xc_train, yc_train)
    yc_pred = classifier.predict(Xc_test)

    classification_metrics = {
        "Accuracy": accuracy_score(yc_test, yc_pred),
        "Precision": precision_score(yc_test, yc_pred, zero_division=0),
        "Recall": recall_score(yc_test, yc_pred, zero_division=0),
        "F1 Score": f1_score(yc_test, yc_pred, zero_division=0)
    }

    # -------------------------
    # Demand regression model
    # -------------------------
    reg_features = [
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

    Xr = df[reg_features].copy()
    yr = df["Demand"]

    categorical_reg = [
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numeric_reg = [c for c in reg_features if c not in categorical_reg]

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_reg),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_reg)
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor_reg),
            ("model", RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr,
        test_size=0.20,
        random_state=42
    )

    regressor.fit(Xr_train, yr_train)
    yr_pred = regressor.predict(Xr_test)

    regression_metrics = {
        "MAE": mean_absolute_error(yr_test, yr_pred),
        "RMSE": np.sqrt(mean_squared_error(yr_test, yr_pred)),
        "R2": r2_score(yr_test, yr_pred)
    }

    return (
        classifier,
        regressor,
        clf_features,
        reg_features,
        classification_metrics,
        regression_metrics
    )


# ---------------------------------------------------------
# LOAD DATA + MODELS
# ---------------------------------------------------------

try:
    df = load_data()
except Exception as e:
    st.error("Dataset load झाला नाही.")
    st.code(f"Expected file: {DATA_PATH}\n\nError: {e}")
    st.stop()

if df["Stockout"].nunique() < 2:
    st.error(
        "Stockout target मध्ये फक्त एक class तयार झाला आहे. "
        "Risk threshold बदलण्याची गरज आहे."
    )
    st.stop()

try:
    (
        classifier,
        regressor,
        clf_features,
        reg_features,
        classification_metrics,
        regression_metrics
    ) = train_models(df)
except Exception as e:
    st.error("Models train झाले नाहीत.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("📦 Smart Inventory AI")
st.sidebar.write("Stockout & Restock Prediction System")

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


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

if page == "🏠 Home":
    st.title("📦 Smart Inventory Stockout & Restock Prediction")

    st.subheader("AI-based inventory decision support system")

    st.write(
        """
        This mini-project uses Machine Learning to analyze retail inventory,
        predict product demand, identify stockout risk and recommend a
        restock quantity.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Products", df["Product ID"].nunique())
    c3.metric("Stores", df["Store ID"].nunique())
    c4.metric("Categories", df["Category"].nunique())

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown(
        """
        - Predict future product demand
        - Detect stockout risk
        - Recommend restock quantity
        - Analyze sales and inventory trends
        - Provide an interactive web dashboard
        """
    )

    st.info(
        "The dataset is a retail inventory/demand dataset. "
        "The stockout target used by this demo is created from historical "
        "inventory and sales using an inventory-coverage rule."
    )


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

elif page == "📊 Dashboard":
    st.title("📊 Inventory Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Units Sold", f"{df['Units Sold'].sum():,.0f}")
    col2.metric("Average Inventory", f"{df['Inventory Level'].mean():,.1f}")
    col3.metric("Average Demand", f"{df['Demand'].mean():,.1f}")
    col4.metric("Stockout Risk Records", f"{df['Stockout'].sum():,}")

    st.divider()

    daily_sales = df.groupby("Date")["Units Sold"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_sales.index, daily_sales.values)
    ax.set_title("Daily Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Category-wise Sales")

    category_sales = (
        df.groupby("Category")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_sales)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

elif page == "🔮 Prediction":
    st.title("🔮 Product Demand & Stockout Prediction")

    st.write("Enter product/store conditions below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        category = st.selectbox("Category", sorted(df["Category"].dropna().unique()))
        region = st.selectbox("Region", sorted(df["Region"].dropna().unique()))
        weather = st.selectbox(
            "Weather Condition",
            sorted(df["Weather Condition"].dropna().unique())
        )

    with col2:
        seasonality = st.selectbox(
            "Seasonality",
            sorted(df["Seasonality"].dropna().unique())
        )
        inventory = st.number_input(
            "Current Inventory",
            min_value=0.0,
            value=float(df["Inventory Level"].median()),
            step=1.0
        )
        price = st.number_input(
            "Price",
            min_value=0.0,
            value=float(df["Price"].median()),
            step=0.01
        )

    with col3:
        discount = st.number_input(
            "Discount",
            min_value=0.0,
            value=float(df["Discount"].median()),
            step=1.0
        )
        promotion = st.selectbox(
            "Promotion",
            sorted(df["Promotion"].dropna().unique())
        )
        competitor_price = st.number_input(
            "Competitor Pricing",
            min_value=0.0,
            value=float(df["Competitor Pricing"].median()),
            step=0.01
        )

    epidemic = st.selectbox(
        "Epidemic",
        sorted(df["Epidemic"].dropna().unique())
    )

    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1,
        max_value=60,
        value=7
    )

    safety_stock = st.number_input(
        "Safety Stock",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    if st.button("🚀 Predict", use_container_width=True):

        current_date = pd.Timestamp.now()

        input_data = pd.DataFrame([{
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
        }])

        predicted_demand = float(regressor.predict(input_data[reg_features])[0])

        # Model-based stockout probability
        stockout_class = int(
            classifier.predict(input_data[clf_features])[0]
        )

        if hasattr(classifier, "predict_proba"):
            probability = float(
                classifier.predict_proba(input_data[clf_features])[0][1]
            )
        else:
            probability = float(stockout_class)

        # Restock recommendation
        target_stock = (predicted_demand * lead_time) + safety_stock
        recommended_restock = max(0.0, target_stock - inventory)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Demand",
            f"{predicted_demand:.0f} units"
        )

        c2.metric(
            "Stockout Risk",
            "HIGH" if stockout_class == 1 else "LOW"
        )

        c3.metric(
            "Recommended Restock",
            f"{recommended_restock:.0f} units"
        )

        st.progress(min(max(probability, 0.0), 1.0))

        st.write(
            f"Estimated stockout probability: **{probability * 100:.1f}%**"
        )

        st.info(
            f"Recommended target stock = predicted demand × lead time + safety stock "
            f"= {predicted_demand:.1f} × {lead_time} + {safety_stock:.1f}."
        )


# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

elif page == "📈 Analytics":
    st.title("📈 Inventory Analytics")

    st.subheader("Inventory Level vs Units Sold")

    sample = df.sample(
        min(3000, len(df)),
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        sample["Inventory Level"],
        sample["Units Sold"],
        alpha=0.35
    )
    ax.set_xlabel("Inventory Level")
    ax.set_ylabel("Units Sold")
    ax.set_title("Inventory Level vs Units Sold")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby(["Year", "Month"])["Units Sold"]
        .sum()
        .reset_index()
    )

    monthly_sales["Period"] = (
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str).str.zfill(2)
    )

    st.line_chart(
        monthly_sales.set_index("Period")["Units Sold"]
    )

    st.subheader("Region-wise Sales")

    region_sales = (
        df.groupby("Region")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(region_sales)


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    st.subheader("Stockout Classification — Random Forest")

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

    st.subheader("Demand Prediction — Random Forest Regressor")

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

    st.success(
        "Models are trained automatically when the Streamlit application starts."
    )

st.sidebar.divider()
st.sidebar.caption("Smart Inventory Stockout & Restock Prediction")
