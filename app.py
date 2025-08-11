import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_first_sheet(uploaded_file):
    """
    Read the uploaded Excel file and return a DataFrame from the most relevant sheet.
    If multiple sheets exist, it picks the first sheet containing a 'sales' or 'amount' column,
    otherwise it returns the first sheet.
    """
    xls = pd.ExcelFile(uploaded_file)
    for sheet_name in xls.sheet_names:
        data = pd.read_excel(xls, sheet_name=sheet_name)
        cols = [str(c).strip().lower() for c in data.columns]
        if any('sales' in c or 'amount' in c for c in cols):
            return data
    # Fall back to the first sheet if no obvious sales sheet is found
    return pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])

def clean_data_generic(df):
    """
    Standardise column names, extract date/product/sales/expenses,
    and compute profit and profit margin.
    """
    df = df.copy()

    # Standardise column names
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]

    # Identify date column
    date_col = None
    for c in df.columns:
        if c == 'date':
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if 'date' in c:
                date_col = c
                break
    # Create date field
    if date_col is not None:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df['date'] = pd.NaT

    # Identify sales column
    sales_col = None
    for c in df.columns:
        if c == 'sales':
            sales_col = c
            break
    if sales_col is None:
        for c in df.columns:
            if 'sales' in c:
                sales_col = c
                break
    if sales_col is None:
        for c in df.columns:
            if 'amount' in c:
                sales_col = c
                break
    if sales_col is not None:
        df['sales'] = pd.to_numeric(df[sales_col], errors='coerce')
    else:
        df['sales'] = np.nan

    # Identify expenses column
    exp_col = None
    for c in df.columns:
        if c == 'expenses' or c == 'expense':
            exp_col = c
            break
    if exp_col is None:
        for c in df.columns:
            if 'expense' in c:
                exp_col = c
                break
    if exp_col is not None:
        df['expenses'] = pd.to_numeric(df[exp_col], errors='coerce')
    else:
        # If no expenses column, set to zero
        df['expenses'] = 0.0

    # Identify product column
    prod_col = None
    for c in df.columns:
        if c == 'product':
            prod_col = c
            break
    if prod_col is None:
        for c in df.columns:
            if 'product' in c:
                prod_col = c
                break
    if prod_col is None:
        for c in df.columns:
            if 'item' in c:
                prod_col = c
                break
    if prod_col is None:
        for c in df.columns:
            if 'category' in c:
                prod_col = c
                break
    if prod_col is not None:
        df['product'] = df[prod_col].astype(str)
    else:
        df['product'] = 'Unknown'

    # Remove rows with invalid dates
    df = df[df['date'].notna()]

    # Replace missing sales/expenses with zero
    df['sales'] = df['sales'].fillna(0.0)
    df['expenses'] = df['expenses'].fillna(0.0)

    # Calculate profit and profit margin
    df['profit'] = df['sales'] - df['expenses']
    df['profit_margin'] = np.where(df['sales'] != 0,
                                   (df['profit'] / df['sales']) * 100,
                                   0.0)

    return df[['date', 'product', 'sales', 'expenses', 'profit', 'profit_margin']]

def create_summaries(df):
    """
    Create daily totals, product totals, and monthly aggregates from cleaned data.
    """
    # Daily totals
    daily = (df.groupby('date')
               .agg({'sales':'sum','expenses':'sum','profit':'sum'})
               .reset_index())
    daily['profit_margin'] = np.where(daily['sales'] != 0,
                                      (daily['profit'] / daily['sales']) * 100,
                                      0.0)

    # Product totals
    product_totals = (df.groupby('product')
                        .agg({'sales':'sum','expenses':'sum','profit':'sum'})
                        .sort_values(by='sales', ascending=False)
                        .reset_index())
    product_totals['profit_margin'] = np.where(product_totals['sales'] != 0,
                                               (product_totals['profit'] / product_totals['sales']) * 100,
                                               0.0)

    # Monthly aggregates
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = (df.groupby('year_month')
                 .agg({'sales':'sum','expenses':'sum','profit':'sum'})
                 .reset_index())
    monthly['profit_margin'] = np.where(monthly['sales'] != 0,
                                        (monthly['profit'] / monthly['sales']) * 100,
                                        0.0)
    # Convert Period to datetime for plotting
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

    return daily, product_totals, monthly

def main():
    st.set_page_config(page_title='Sales and Expenses Dashboard', layout='wide')
    st.title('Sales and Expenses Dashboard')

    uploaded_file = st.file_uploader('Upload Excel file', type=['xls','xlsx'])

    if uploaded_file is None:
        st.info('Please upload an Excel file to continue.')
        st.stop()

    # Load and clean the data
    raw_df = load_first_sheet(uploaded_file)
    df = clean_data_generic(raw_df)

    if df.empty:
        st.warning('No valid data was found in the uploaded file.')
        st.stop()

    # Summaries
    daily_totals, product_totals, monthly_totals = create_summaries(df)

    # KPIs
    total_sales = df['sales'].sum()
    total_expenses = df['expenses'].sum()
    total_profit = df['profit'].sum()
    avg_profit_margin = (df.loc[df['sales'] != 0, 'profit_margin']).mean()

    # Display KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric('Total Sales', f"${total_sales:,.2f}")
    kpi_col2.metric('Total Expenses', f"${total_expenses:,.2f}")
    kpi_col3.metric('Total Profit', f"${total_profit:,.2f}")
    kpi_col4.metric('Average Profit Margin', f"{avg_profit_margin:.2f}%")

    st.markdown('---')

    # Line chart of daily sales and profit over time
    fig_line = px.line(daily_totals,
                       x='date',
                       y=['sales','profit'],
                       labels={'value':'Amount','variable':'Metric','date':'Date'},
                       title='Daily Sales and Profit Over Time')
    fig_line.update_layout(legend_title_text='', xaxis_title='Date', yaxis_title='Amount')
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart of top products by sales
    top_products = product_totals.nlargest(10, 'sales')
    fig_bar = px.bar(top_products,
                     x='product',
                     y='sales',
                     labels={'product':'Product','sales':'Sales'},
                     title='Top Products by Sales')
    fig_bar.update_layout(xaxis_title='Product', yaxis_title='Sales')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Raw data table with simple filters
    st.markdown('### Raw Data')
    with st.expander('Show filters'):
        # Date range filter
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.date_input('Select date range:',
                                   value=(min_date, max_date),
                                   min_value=min_date,
                                   max_value=max_date)
        # Product filter
        products_selected = st.multiselect('Select products to display:',
                                           options=sorted(df['product'].unique()),
                                           default=sorted(df['product'].unique())[:10])
    # Apply filters
    filtered_df = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    if products_selected:
        filtered_df = filtered_df[filtered_df['product'].isin(products_selected)]
    st.dataframe(filtered_df.sort_values(by='date'), use_container_width=True)

    st.markdown('---')