import numpy as np
import pandas as pd

CATEGORICAL = [
    "Country", "Managed By", "Fulfill Via", "Vendor INCO Term", "Shipment Mode", "Product Group",
    "Sub Classification", "Vendor", "Molecule/Test Type", "Brand", "Dosage Form", "Manufacturing Site",
    "First Line Designation", "Project Code", "Item Description", "Dosage", "weight_status", "freight_status",
]
DATE_FMT = {"Scheduled Delivery Date": "%d-%b-%y", "Delivered to Client Date": "%d-%b-%y",
            "PQ First Sent to Client Date": "%m/%d/%y", "PO Sent to Vendor Date": "%m/%d/%y"}


def load(path="SCMS_Delivery_History_Dataset.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    for c, f in DATE_FMT.items():
        df[c] = pd.to_datetime(df[c], format=f, errors="coerce")
    df["days_late"] = (df["Delivered to Client Date"] - df["Scheduled Delivery Date"]).dt.days
    df["is_late"] = (df["days_late"] > 0).astype(int)
    return df.sort_values(["Scheduled Delivery Date", "ID"]).reset_index(drop=True)


def _status(s):
    num = pd.to_numeric(s, errors="coerce")
    txt = s.where(num.isna(), "numeric").astype(str).str.replace(r"See (ASN|DN).*", "linked", regex=True)
    return num, txt


def build(df):
    X = df[[c for c in CATEGORICAL if c in df]].copy()
    X["weight"], X["weight_status"] = _status(df["Weight (Kilograms)"].astype(str))
    X["freight"], X["freight_status"] = _status(df["Freight Cost (USD)"].astype(str))
    for c in ["Line Item Quantity", "Line Item Value", "Pack Price", "Unit Price", "Unit of Measure (Per Pack)",
              "Line Item Insurance (USD)"]:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["log_value"] = np.log1p(X["Line Item Value"])
    X["value_per_kg"] = X["Line Item Value"] / X["weight"].replace(0, np.nan)
    X["freight_ratio"] = X["freight"] / X["Line Item Value"].replace(0, np.nan)
    X["insurance_ratio"] = X["Line Item Insurance (USD)"] / X["Line Item Value"].replace(0, np.nan)
    sched, pq, po = df["Scheduled Delivery Date"], df["PQ First Sent to Client Date"], df["PO Sent to Vendor Date"]
    X["sched_year"], X["sched_month"] = sched.dt.year, sched.dt.month
    X["sched_dow"], X["sched_doy"] = sched.dt.dayofweek, sched.dt.dayofyear
    X["sched_ord"] = (sched - pd.Timestamp("2006-01-01")).dt.days
    X["pq_to_sched"] = (sched - pq).dt.days
    X["po_to_sched"] = (sched - po).dt.days
    X["pq_to_po"] = (po - pq).dt.days
    X["is_pre_pq"] = df["PQ #"].eq("Pre-PQ Process").astype(int)
    X["lines_in_shipment"] = df.groupby("ASN/DN #")["ID"].transform("size")
    X["lines_in_po"] = df.groupby("PO / SO #")["ID"].transform("size")
    X["shipment_value"] = df.groupby("ASN/DN #")["Line Item Value"].transform("sum")
    X["country_daily_load"] = df.groupby(["Country", "Scheduled Delivery Date"])["ID"].transform("size")
    X["vendor_daily_load"] = df.groupby(["Vendor", "Scheduled Delivery Date"])["ID"].transform("size")
    X["po_seq"] = pd.to_numeric(df["PO / SO #"].str.extract(r"(\d+)")[0], errors="coerce")
    X["asn_seq"] = pd.to_numeric(df["ASN/DN #"].str.extract(r"(\d+)")[0], errors="coerce")
    for c in [c for c in CATEGORICAL if c in X]:
        X[c] = X[c].fillna("missing").astype(str).astype("category")
    return X
