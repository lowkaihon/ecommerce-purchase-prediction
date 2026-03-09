import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CATEGORICAL_FEATURES, RANDOM_STATE, TEST_SIZE


def clean_data(df):
    df = df.copy()
    df["CustomerType"] = df["CustomerType"].replace(
        "returning_Visitor", "Returning_Visitor"
    )
    df["CustomerType"] = df["CustomerType"].apply(
        lambda x: "Unknown" if x in ("", "nan", "None") else x
    )
    df["GeographicRegion"] = df["GeographicRegion"].apply(
        lambda x: abs(x) if x < 0 else x
    )
    df["BounceRate"] = df["BounceRate"].apply(lambda x: abs(x) if x < 0 else x)
    df["ProductPageTime"] = df["ProductPageTime"].apply(
        lambda x: abs(x) if x < 0 else x
    )
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def feature_engineer(df):
    df = df.copy()
    df["has_page_value"] = (df["PageValue"] > 0).astype(int)
    df["PageValue_log"] = np.log1p(df["PageValue"])
    df["ProductPageTime_log"] = np.log1p(df["ProductPageTime"])
    return df


def prepare_data(df):
    df = df.copy()
    # Convert numeric categoricals to string for one-hot encoding
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    df_encoded = pd.get_dummies(
        df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int
    )
    X = df_encoded.drop(columns="PurchaseCompleted")
    y = df_encoded["PurchaseCompleted"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
