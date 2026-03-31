import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_cleaning import clean_customers_data

st.set_page_config(page_title='Customer Purchase Platform', layout='wide')

# -------------------------------
# Paths
# -------------------------------
BASE = Path(__file__).resolve().parents[1]
data_dir = BASE / 'data'
models_dir = BASE / 'code'/'models'

# -------------------------------
# Load Datasets
# -------------------------------
@st.cache_data
def load_data():
    customers = pd.read_csv(data_dir / 'customers.csv')
    products = pd.read_csv(data_dir / 'ecommerce_products_110.csv')
    transactions = pd.read_csv(data_dir / 'transactions.csv')
    return customers, products, transactions

customers, products, transactions = load_data()

# -------------------------------
# Data Cleaning
# -------------------------------
@st.cache_data
def get_clean_customers(df):
    return clean_customers_data(df)

customers_clean = get_clean_customers(customers)

# -------------------------------
# Load ML Models
# -------------------------------
try:
    logistic = joblib.load(models_dir / 'model_logistic.pkl')
    rf = joblib.load(models_dir / 'model_randomforest.pkl')
    xgb = joblib.load(models_dir / 'model_xgboost.pkl')
except Exception as e:
    st.warning(f"⚠ Model loading error: {e}")
    logistic = rf = xgb = None

# -------------------------------
# Load Supporting Files
# -------------------------------
rfm = pd.read_csv(models_dir / 'rfm_customers.csv') if (models_dir / 'rfm_customers.csv').exists() else None
pop = pd.read_csv(models_dir / 'product_popularity.csv') if (models_dir / 'product_popularity.csv').exists() else None

try:
    sim_df = pd.read_csv(models_dir / 'item_similarity.csv', index_col=0)
except:
    sim_df = None

# -------------------------------
# Streamlit Setup
# -------------------------------

st.title('Customer Purchase Prediction + RFM + Segmentation + Recommendations')

page = st.sidebar.radio(
    'Go to',
    ['Predict',
     'Customer Scoring',
     'RFM & Segments',
     'Recommendations',
     'EDA & Visualizations',
     'Feature Importance',
     'Dataset']
)

# ==========================================================
# PAGE 1: Purchase Prediction
# ==========================================================
if page == 'Predict':
    st.header('Purchase Prediction')

    if logistic is None or rf is None or xgb is None:
        st.error("Models not found.")
        st.stop()

    model_choice = st.selectbox(
        'Select Model',
        ['LogisticRegression', 'RandomForest', 'XGBoost']
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', 18, 100, 30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        avg_browsing_time = st.number_input('Average Browsing Time', 0.1, 300.0, 30.0)

    with col2:
        avg_page_views = st.number_input('Average Page Views', 1, 100, 10)
        past_purchases = st.number_input('Past Purchases', 0, 100, 5)

    if st.button('Predict Purchase'):
        X = pd.DataFrame([{
            'Age': age,
            'Gender': 0 if gender == 'Male' else 1,
            'AvgBrowsingTime': avg_browsing_time,
            'AvgPageViews': avg_page_views,
            'PastPurchases': past_purchases
        }])

        model = logistic if model_choice == 'LogisticRegression' else rf if model_choice == 'RandomForest' else xgb
        prob = model.predict_proba(X)[0][1]
        pred = 1 if prob >= 0.5 else 0

        if pred:
            st.success(f'Customer WILL PURCHASE (Probability: {prob:.2%})')
        else:
            st.error(f'Customer WILL NOT PURCHASE (Probability: {prob:.2%})')

        st.json({
            "Age": age,
            "Gender": gender,
            "AvgBrowsingTime": avg_browsing_time,
            "AvgPageViews": avg_page_views,
            "PastPurchases": past_purchases,
            "Model Used": model_choice
        })

# ==========================================================
# PAGE 2: CUSTOMER SCORING
# ==========================================================
elif page == 'Customer Scoring':

    st.header("Customer Purchase Propensity Scoring")

    scored_path = models_dir / "scored_customers.csv"

    if not scored_path.exists():
        st.error("Scored customer file not found. Run training first.")
    else:
        scored_df = pd.read_csv(scored_path)

        st.subheader("Top 10 High Probability Customers")
        top10 = scored_df.sort_values(
            "Purchase_Probability",
            ascending=False
        ).head(10)

        st.dataframe(top10)

        st.subheader("Top 20% Target Customers")

        top20 = scored_df[scored_df["Target_Top20"] == 1]
        st.write(f"Total Target Customers: {len(top20)}")

        st.dataframe(top20.head(20))

        # Probability distribution
        st.subheader("Probability Distribution")

        fig, ax = plt.subplots()
        ax.hist(scored_df["Purchase_Probability"], bins=20)
        ax.set_title("Purchase Probability Distribution")
        st.pyplot(fig)

        # Download option
        st.download_button(
            label="Download Scored Customers CSV",
            data=scored_df.to_csv(index=False),
            file_name="scored_customers.csv",
            mime="text/csv"
        )

# ==========================================================
# PAGE 3: RFM & Segmentation
# ==========================================================
elif page == 'RFM & Segments':
    st.header('RFM Analysis & Segments')

    if rfm is None:
        st.error("RFM data not available.")
    else:
        st.dataframe(rfm.sample(10))
        seg = st.selectbox('Select Cluster', sorted(rfm['Cluster'].unique()))
        st.dataframe(rfm[rfm['Cluster'] == seg].head(20))

# ==========================================================
# PAGE 4: Recommendations
# ==========================================================
elif page == 'Recommendations':

    st.header('Customer Product Insights')

    cust = st.selectbox(
        'Choose Customer ID',
        transactions['CustomerID'].unique()
    )

    # -------------------------------
    # 🔹 TOP POPULAR PRODUCTS
    # -------------------------------
    st.subheader('Top Popular Products')

    top_popular = (
        transactions
        .groupby('ProductID')['Quantity']
        .sum()
        .reset_index(name='TotalPurchased')
        .merge(products, on='ProductID', how='left')
        .sort_values('TotalPurchased', ascending=False)
        .head(10)
    )

    st.table(
        top_popular[['ProductID', 'ProductName', 'Category', 'Price', 'TotalPurchased']]
        .rename(columns={
            'ProductID': 'Product ID',
            'ProductName': 'Product Name',
            'Category': 'Category',
            'Price': 'Price',
            'TotalPurchased': 'Total Quantity Sold'
        })
    )

    # -------------------------------
    # 🔹 CUSTOMER PURCHASE HISTORY
    # -------------------------------
    st.subheader(f'Customer {cust} – Purchase History')

    customer_tx = transactions[transactions['CustomerID'] == cust]

    if customer_tx.empty:
        st.warning('No purchase record found for this customer.')
    else:

        # Merge with product details
        customer_data = (
            customer_tx
            .groupby('ProductID')['Quantity']
            .sum()
            .reset_index(name='Purchased Quantity')
            .merge(products, on='ProductID', how='left')
            .sort_values('Purchased Quantity', ascending=False)
        )

        st.table(
            customer_data[['ProductID', 'ProductName', 'Category', 'Purchased Quantity']]
            .rename(columns={
                'ProductID': 'Product ID',
                'ProductName': 'Product Name',
                'Category': 'Category',
            })
        )

        # -------------------------------
        # TOTAL CUSTOMER SPEND
        # -------------------------------
        total_spend = customer_tx['TotalAmount'].sum()

        # -------------------------------
        #  FAVORITE CATEGORY
        # -------------------------------
        category_pref = (
            customer_tx
            .merge(products, on='ProductID', how='left')
            .groupby('Category')['Quantity']
            .sum()
            .sort_values(ascending=False)
        )

        favorite_category = category_pref.index[0]

        # Display metrics
        col1, col2 = st.columns(2)

        col1.metric("Total Spend", f"₹ {total_spend:,.2f}")
        col2.metric("Favorite Category", favorite_category)


# ==========================================================
# PAGE 5: EDA & Visualizations
# ==========================================================
elif page == 'EDA & Visualizations':
    st.header('Exploratory Data Analysis')

    # Purchase Distribution
    fig, ax = plt.subplots()
    sns.countplot(x='Purchase', data=customers_clean, ax=ax)
    ax.set_title('Purchase vs No Purchase')
    st.pyplot(fig)

    # Gender vs Purchase
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='Purchase', data=customers_clean, ax=ax)
    ax.set_title('Gender vs Purchase')
    st.pyplot(fig)

    # Browsing Time Distribution
    fig, ax = plt.subplots()
    ax.hist(customers_clean['AvgBrowsingTime'], bins=20)
    ax.set_title('Browsing Time Distribution')
    st.pyplot(fig)

    # Boxplot
    fig, ax = plt.subplots()
    sns.boxplot(x='Purchase', y='AvgBrowsingTime', data=customers_clean, ax=ax)
    ax.set_title('Browsing Time vs Purchase')
    st.pyplot(fig)

    # Scatter Plot
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='AvgPageViews',
        y='AvgBrowsingTime',
        hue='Purchase',
        data=customers_clean,
        ax=ax
    )
    ax.set_title('Engagement vs Purchase')
    st.pyplot(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    corr = customers_clean.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

# ==========================================================
# PAGE 6: Feature Importance
# ==========================================================
elif page == "Feature Importance":

    st.header("Feature Importance Analysis")

    rf = joblib.load(BASE /"code"/ "models" / "model_randomforest.pkl")

    # Access internal RandomForest (new sklearn compatible)
    rf_pipeline = rf.calibrated_classifiers_[0].estimator
    rf_model = rf_pipeline.named_steps["clf"]

    importances = rf_model.feature_importances_

    feature_names = [
        "Age",
        "Gender",
        "AvgBrowsingTime",
        "AvgPageViews",
        "PastPurchases"
    ]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

    st.info("This shows which features most influence purchase decisions.")



# ==========================================================
# PAGE 7: Dataset Preview
# ==========================================================
elif page == 'Dataset':
    st.header('Dataset Preview')
    st.write('Cleaned Customer Data')
    st.dataframe(customers_clean.sample(10))
