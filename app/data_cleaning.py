import pandas as pd

def clean_customers_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses customer dataset.
    This function is reused across EDA, ML, and Streamlit.
    """

    df = df.copy()

    # -----------------------------
    # Remove duplicates
    # -----------------------------
    df.drop_duplicates(inplace=True)

    # -----------------------------
    # Standardize Gender
    # -----------------------------
    df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
    df = df[df['Gender'].isin(['Male', 'Female'])]

    # -----------------------------
    # Convert numeric columns safely
    # -----------------------------
    numeric_cols = [
        'Age',
        'AvgBrowsingTime',
        'AvgPageViews',
        'PastPurchases',
        'Purchase'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # -----------------------------
    # Handle missing values
    # -----------------------------
    df.fillna({
        'Age': df['Age'].median(),
        'AvgBrowsingTime': df['AvgBrowsingTime'].median(),
        'AvgPageViews': df['AvgPageViews'].median(),
        'PastPurchases': df['PastPurchases'].median(),
        'Purchase': 0
    }, inplace=True)

    # -----------------------------
    # Filter unrealistic values
    # -----------------------------
    df = df[(df['Age'] >= 18) & (df['Age'] <= 80)]
    df = df[df['AvgBrowsingTime'] > 0]
    df = df[df['AvgPageViews'] > 0]
    df = df[df['PastPurchases'] >= 0]

    return df
