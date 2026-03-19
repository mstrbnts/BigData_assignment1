# Import needed packages
import pandas as pd
import numpy as np
from config import ORIG

# Import data
df_tx = pd.read_csv(
    "data/original_data/data/transactions_2016_2017.csv",
    low_memory=False
)
pd.set_option('display.max_columns', None)


# Feature pipeline
def build_customer_features(df):
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["pack_date"] = pd.to_datetime(df["pack_date"], errors="coerce")
    df["sale_revenue"] = pd.to_numeric(df["sale_revenue"], errors="coerce")
    df["sale_discount_applied"] = pd.to_numeric(df["sale_discount_applied"], errors="coerce")
    df["prod_size"] = pd.to_numeric(df["prod_size"], errors="coerce")


    # Dates and returns
    df["order_date"] = pd.to_datetime(df["order_date"])
    if "is_returned" not in df.columns:
        if "returned_to_shop_id" in df.columns:
            df["is_returned"] = df["returned_to_shop_id"].notna().astype(int)
        else:
            df["is_returned"] = 0

    cutoff_date = df["order_date"].max() + pd.Timedelta(days=1)

    # Only non-returned items for discount and revenue featues
    df_non_returned = df[df["is_returned"] == 0]

    # Basic features
    agg = df.groupby("cust_id").agg(
        n_items=("sale_id", "count"),
        n_sales=("sale_id", "nunique"),
        n_returns=("is_returned", "sum"),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
    )

    # Returned items/sales
    agg["n_items_non_returned"] = df_non_returned.groupby("cust_id")["sale_id"].count()
    agg["n_sales_non_returned"] = df_non_returned.groupby("cust_id")["sale_id"].nunique()

    # Revenue (net, gross and discount)
    agg["total_revenue_net"] = df_non_returned.groupby("cust_id")["sale_revenue"].sum()
    agg["total_discount"] = df_non_returned.groupby("cust_id")["sale_discount_applied"].apply(lambda s: s.abs().sum())
    agg["total_revenue_gross"] = agg["total_revenue_net"] + agg["total_discount"]

    # Derived features
    agg["customer_lifetime_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    agg["recency_days"] = (cutoff_date - agg["last_purchase"]).dt.days
    agg["avg_rev_per_item"] = agg["total_revenue_net"] / agg["n_items_non_returned"]
    agg["avg_rev_per_sale"] = agg["total_revenue_net"] / agg["n_sales_non_returned"]
    agg["return_rate"] = agg["n_returns"] / agg["n_items"]
    agg["discount_ratio"] = agg["total_discount"] / (agg["total_revenue_net"] + agg["total_discount"])

    # Brands 
    brand_counts = df.groupby(["cust_id", "prod_brand"]).size().unstack(fill_value=0)
    brand_share = brand_counts.div(brand_counts.sum(axis=1), axis=0)
    agg["top_brand_share"] = brand_share.max(axis=1)

    # Web-only
    agg["share_web_only"] = df.groupby("cust_id")["prod_web_only"].mean()

    # Shoesize
    size_stats = df.groupby("cust_id")["prod_size"].agg(
        size_mean="mean",
        size_std=lambda s: s.std(ddof=0),  # ddof=0 -> std=0 bij 1 observatie
        size_min="min",
        size_max="max",
    )
    agg = agg.join(size_stats)
    agg["size_std"] = agg["size_std"].fillna(0)

    # Season (Winter/Summer + NOS/CONS)
    if "season" in df.columns:
        s = df["season"].astype(str).str.strip().str.lower()
        is_summer = s.eq("summer")
        is_winter = s.eq("winter")
        is_other = s.isin(["nos", "cons"])  # NOS / CONS

        agg["share_summer_items"] = df.assign(_is_summer=is_summer).groupby("cust_id")["_is_summer"].mean()
        agg["share_winter_items"] = df.assign(_is_winter=is_winter).groupby("cust_id")["_is_winter"].mean()
        agg["share_other_season_items"] = df.assign(_is_other=is_other).groupby("cust_id")["_is_other"].mean()

    # Season type shares (main, mid-season, special, other) 
    if "season_type" in df.columns:
        st = df["season_type"].astype(str).str.strip().str.lower()
        agg["share_season_type_main"] = df.assign(_st_main=st.eq("main")).groupby("cust_id")["_st_main"].mean()
        agg["share_season_type_mid_season"] = df.assign(_st_mid=st.eq("mid-season")).groupby("cust_id")["_st_mid"].mean()
        agg["share_season_type_special"] = df.assign(_st_special=st.eq("special")).groupby("cust_id")["_st_special"].mean()
        agg["share_season_type_other"] = df.assign(_st_other=st.eq("other")).groupby("cust_id")["_st_other"].mean()

    # Luxery proxy (price quantile per item) ---
    if "revenue_quantile" in df.columns:
        high_map = {"Q5 (80-100%)": 1, "Q4 (60-80%)": 1}
        df["is_high_revenue_item"] = df["revenue_quantile"].map(high_map).fillna(0).astype(int)
        agg["share_high_revenue_items"] = df.groupby("cust_id")["is_high_revenue_item"].mean()

    # Material
    if "material_main" in df.columns:
        mat_main_counts = df.groupby(["cust_id", "material_main"]).size().unstack(fill_value=0)
        mat_main_share = mat_main_counts.div(mat_main_counts.sum(axis=1), axis=0)
        agg["material_main_dominant_share"] = mat_main_share.max(axis=1)
        agg["material_main_n_unique"] = (mat_main_counts > 0).sum(axis=1)

    return agg.reset_index()

# Implement features
features = build_customer_features(df_tx)
print("Features are made")

# Run to get csv file with features, uncomment to get a csv
features.to_csv("data/features.csv", index=False)