import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_first_sheet(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        cols = [str(c).lower() for c in df.columns]
        if any("sales" in c or "amount" in c for c in cols):
            return df
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def clean(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # date
    date_col = next((c for c in df.columns if c == "date" or "date" in c), None)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    # product
    prod_col = next((c for c in df.columns if c in ["product"] or "item" in c or "category" in c), None)
    df["product"] = df[prod_col].astype(str) if prod_col else "Unknown"
    # sales
    sales_col = next((c for c in df.columns if c == "sales" or "sales" in c or "amount" in c), None)
    df["sales"] = pd.to_numeric(df[sales_col], errors="coerce") if sales_col else 0.0
    # expenses
    exp_col = next((c for c in df.columns if c.startswith("expense")), None)
    df["expenses"] = pd.to_numeric(df[exp_col], errors="coerce") if exp_col else 0.0
    df = df[df["date"].notna()]
    df["sales"] = df["sales"].fillna(0.0)
    df["expenses"] = df["expenses"].fillna(0.0)
    df["profit"] = df["sales"] - df["expenses"]
    df["profit_margin"] = np.where(df["sales"] != 0, (df["profit"]/df["sales"])*100, 0.0)
    return df[["date","product","sales","expenses","profit","profit_margin"]]

def main():
    st.set_page_config(page_title="Sales Dashboard", layout="wide")
    st.title("Sales & Expenses Dashboard")

    # ← Uploader is ALWAYS visible, first thing
    uploaded_file = st.file_uploader("Upload Excel file", type=["xls","xlsx"])
    if uploaded_file is None:
        st.info("Upload an Excel file to continue.")
        st.stop()

    raw = load_first_sheet(uploaded_file)
    df = clean(raw)

    st.success(f"Loaded {len(df):,} rows from file.")
    st.dataframe(df.head(20), use_container_width=True)

if __name__ == "__main__":
    main()
