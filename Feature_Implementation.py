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

    # Normalize boolean-like columns if needed
    for col in ["prod_web_only", "prod_outlet"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            )

    # Returns
    if "is_returned" not in df.columns:
        if "returned_to_shop_id" in df.columns:
            df["is_returned"] = df["returned_to_shop_id"].notna().astype(int)
        else:
            df["is_returned"] = 0

    # Shipping timing
    if "pack_date" in df.columns:
        df["days_to_pack"] = (df["pack_date"] - df["order_date"]).dt.days

    cutoff_date = df["order_date"].max() + pd.Timedelta(days=1)

    # Non-returned items only
    df_non_returned = df[df["is_returned"] == 0].copy()

    # Basic features
    agg = df.groupby("cust_id").agg(
        n_items=("sale_id", "count"),
        n_sales=("sale_id", "nunique"),
        n_returns=("is_returned", "sum"),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
    )

    # Returned / non-returned counts
    agg["n_items_non_returned"] = df_non_returned.groupby("cust_id")["sale_id"].count()
    agg["n_sales_non_returned"] = df_non_returned.groupby("cust_id")["sale_id"].nunique()

    # Revenue features
    agg["total_revenue_net"] = df_non_returned.groupby("cust_id")["sale_revenue"].sum()
    agg["total_discount"] = df_non_returned.groupby("cust_id")["sale_discount_applied"].apply(lambda s: s.abs().sum())
    agg["total_revenue_gross"] = agg["total_revenue_net"] + agg["total_discount"]

    # Derived features
    agg["customer_lifetime_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    agg["recency_days"] = (cutoff_date - agg["last_purchase"]).dt.days
    agg["avg_rev_per_item"] = agg["total_revenue_net"] / agg["n_items_non_returned"].replace(0, np.nan)
    agg["avg_rev_per_sale"] = agg["total_revenue_net"] / agg["n_sales_non_returned"].replace(0, np.nan)
    agg["return_rate"] = agg["n_returns"] / agg["n_items"].replace(0, np.nan)
    agg["discount_ratio"] = agg["total_discount"] / (
        agg["total_revenue_net"] + agg["total_discount"]
    ).replace(0, np.nan)

    # @Dorian
    # Purchase frequency
    agg["purchase_frequency"] = agg["n_sales_non_returned"] / (agg["customer_lifetime_days"] + 1)
    agg["items_per_sale"] = agg["n_items_non_returned"] / agg["n_sales_non_returned"].replace(0, np.nan)

    order_dates = (
        df_non_returned[["cust_id", "order_date"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["cust_id", "order_date"])
    )

    order_dates["prev_order_date"] = order_dates.groupby("cust_id")["order_date"].shift(1)
    order_dates["days_between_orders"] = (
        order_dates["order_date"] - order_dates["prev_order_date"]
    ).dt.days

    gap_stats = order_dates.groupby("cust_id")["days_between_orders"].agg(
        avg_days_between_orders="mean",
        std_days_between_orders=lambda s: s.std(ddof=0),
        min_days_between_orders="min",
        max_days_between_orders="max"
    )
    agg = agg.join(gap_stats)

    # @Dorian
    # Pack and shipping timing
    if "days_to_pack" in df.columns:
        pack_stats = df_non_returned.groupby("cust_id")["days_to_pack"].agg(
            avg_days_to_pack="mean",
            std_days_to_pack=lambda s: s.std(ddof=0),
            max_days_to_pack="max"
        )
        agg = agg.join(pack_stats)

    # Brands
    if "prod_brand" in df.columns:
        brand_counts = df.groupby(["cust_id", "prod_brand"]).size().unstack(fill_value=0)
        brand_share = brand_counts.div(brand_counts.sum(axis=1), axis=0)
        agg["top_brand_share"] = brand_share.max(axis=1)
        agg["n_unique_brands"] = (brand_counts > 0).sum(axis=1)

    # @Dorian
    # Product diversity
    if "prod_id" in df.columns:
        agg["n_unique_products"] = df.groupby("cust_id")["prod_id"].nunique()

    if "prod_color" in df.columns:
        color_counts = df.groupby(["cust_id", "prod_color"]).size().unstack(fill_value=0)
        color_share = color_counts.div(color_counts.sum(axis=1), axis=0)
        agg["top_color_share"] = color_share.max(axis=1)
        agg["n_unique_colors"] = (color_counts > 0).sum(axis=1)

    for col in ["prod_type_1", "prod_type_3", "prod_type_4", "prod_type_5"]:
        if col in df.columns:
            type_counts = df.groupby(["cust_id", col]).size().unstack(fill_value=0)
            type_share = type_counts.div(type_counts.sum(axis=1), axis=0)
            agg[f"{col}_dominant_share"] = type_share.max(axis=1)
            agg[f"{col}_n_unique"] = (type_counts > 0).sum(axis=1)

    # Web-only
    if "prod_web_only" in df.columns:
        agg["share_web_only"] = df.groupby("cust_id")["prod_web_only"].mean()
        agg["share_web_only_non_returned"] = df_non_returned.groupby("cust_id")["prod_web_only"].mean()

    # Shoe size
    size_stats = df.groupby("cust_id")["prod_size"].agg(
        size_mean="mean",
        size_std=lambda s: s.std(ddof=0),
        size_min="min",
        size_max="max",
        size_median="median",
        size_n_unique="nunique"
    )
    agg = agg.join(size_stats)
    agg["size_std"] = agg["size_std"].fillna(0)

    # @Dorian
    # Season preference
    if "prod_season" in df.columns:
        season_counts = df.groupby(["cust_id", "prod_season"]).size().unstack(fill_value=0)
        season_share = season_counts.div(season_counts.sum(axis=1), axis=0)
        agg["top_season_share"] = season_share.max(axis=1)
        agg["n_unique_seasons"] = (season_counts > 0).sum(axis=1)

    # @Dorian
    # Outlet behavior
    if "prod_outlet" in df.columns:
        agg["share_outlet"] = df.groupby("cust_id")["prod_outlet"].mean()
        agg["share_outlet_non_returned"] = df_non_returned.groupby("cust_id")["prod_outlet"].mean()

    # @Dorian
    # Discount behavior
    df_non_returned["discount_abs"] = df_non_returned["sale_discount_applied"].abs()
    df_non_returned["discounted_item"] = (df_non_returned["discount_abs"] > 0).astype(int)

    discount_stats = df_non_returned.groupby("cust_id").agg(
        avg_discount_per_item=("discount_abs", "mean"),
        max_discount_per_item=("discount_abs", "max"),
        share_discounted_items=("discounted_item", "mean")
    )
    agg = agg.join(discount_stats)

    sale_level = (
        df_non_returned.groupby(["cust_id", "sale_id"])
        .agg(
            sale_revenue_sum=("sale_revenue", "sum"),
            sale_discount_sum=("discount_abs", "sum"),
            sale_items=("sale_id", "count")
        )
        .reset_index()
    )

    sale_level["sale_discount_ratio"] = sale_level["sale_discount_sum"] / (
        sale_level["sale_revenue_sum"] + sale_level["sale_discount_sum"]
    ).replace(0, np.nan)

    sale_stats = sale_level.groupby("cust_id").agg(
        avg_discount_per_sale=("sale_discount_sum", "mean"),
        avg_sale_discount_ratio=("sale_discount_ratio", "mean"),
        avg_items_per_sale=("sale_items", "mean")
    )
    agg = agg.join(sale_stats)

    # @Dorian
    # Returns behavior
    if "prod_brand" in df.columns:
        returned_brand = df[df["is_returned"] == 1].groupby("cust_id")["prod_brand"].nunique()
        agg["n_brands_returned"] = returned_brand

    returned_rev = df[df["is_returned"] == 1].groupby("cust_id")["sale_revenue"].sum()
    agg["returned_revenue_sum"] = returned_rev.abs()

    # @Dorian
    # Product attribute preferences
    for col in [
        "prod_heel",
        "prod_material",
        "prod_insole",
        "prod_print",
        "prod_comfort_sole",
        "prod_comfort_wear",
        "prod_clasp"
    ]:
        if col in df.columns:
            counts = df.groupby(["cust_id", col]).size().unstack(fill_value=0)
            shares = counts.div(counts.sum(axis=1), axis=0)
            agg[f"{col}_dominant_share"] = shares.max(axis=1)
            agg[f"{col}_n_unique"] = (counts > 0).sum(axis=1)

    # Fill numeric missing values
    numeric_cols = agg.select_dtypes(include=[np.number]).columns
    agg[numeric_cols] = agg[numeric_cols].fillna(0)

    return agg.reset_index()

# Implement features
features = build_customer_features(df_tx)
print("Features are made")

# Run to get csv file with features, uncomment to get a csv
features.to_csv("data/features.csv", index=False)