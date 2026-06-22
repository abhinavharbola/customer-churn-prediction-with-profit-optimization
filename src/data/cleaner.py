import pandas as pd
import numpy as np
from config import RAW_DATA_PATH, RANDOM_SEED

def load_raw_data():
    df_0910 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2009-2010", engine="openpyxl")
    df_1011 = pd.read_excel(RAW_DATA_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
    df = pd.concat([df_0910, df_1011], ignore_index=True)
    return df

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)

    # Parse dates
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # Separate cancellations and valid transactions
    df["is_cancellation"] = df["invoice"].astype(str).str.startswith("C")
    cancellations = df[df["is_cancellation"]].copy()
    transactions = df[~df["is_cancellation"]].copy()

    # Remove negative quantities from valid transactions
    transactions = transactions[transactions["quantity"] > 0]

    # Fuzzy match cancellations to valid invoices
    cancellations["matched_invoice"] = cancellations["invoice"].astype(str).str.replace("^C", "", regex=True)
    cancellations["quantity"] = cancellations["quantity"].abs()

    # Merge cancellations with transactions to net out quantities
    cancellation_matches = cancellations.merge(
        transactions[["invoice", "customer_id", "stockcode"]],
        left_on=["matched_invoice", "customer_id", "stockcode"],
        right_on=["invoice", "customer_id", "stockcode"],
        how="inner",
        suffixes=("_cancel", "_trans")
    )

    if not cancellation_matches.empty:
        matched_indices = cancellation_matches["invoice_trans"].values
        matched_quantities = cancellation_matches.groupby("invoice_trans")["quantity_cancel"].sum()
        transactions = transactions.set_index("invoice")
        transactions.loc[matched_indices, "quantity"] -= matched_quantities
        transactions = transactions[transactions["quantity"] > 0].reset_index()

    df_clean = transactions.copy()

    # Create monetary column
    df_clean["revenue"] = df_clean["quantity"] * df_clean["price"]

    # Remove rows with zero or negative price
    df_clean = df_clean[df_clean["price"] > 0]

    # Sort by date
    df_clean = df_clean.sort_values(["customer_id", "invoicedate"]).reset_index(drop=True)

    return df_clean

def run_cleaning():
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    return df_clean