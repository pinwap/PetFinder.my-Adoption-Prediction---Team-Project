import pandas as pd
from sklearn.preprocessing import LabelEncoder

def build_features(df):
    df = df.copy()

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(exclude='object').columns

    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df
