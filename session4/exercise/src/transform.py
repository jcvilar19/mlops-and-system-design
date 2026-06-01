import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import (
    COLUMNS_TO_DROP,
    BINARY_FEATURES,
    ONE_HOT_ENCODE_COLUMNS,
)


class Transformer:
    def __init__(self):
        self.columns_to_drop = COLUMNS_TO_DROP
        self.binary_features = BINARY_FEATURES
        self.one_hot_encode_columns = ONE_HOT_ENCODE_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna()
        df = df.drop(columns=self.columns_to_drop, errors="ignore")
        df = self._preprocess_binary_variables(df)
        df = self._one_hot_encoding(df)
        return df

    def _preprocess_binary_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.binary_features:
            df[col] = df[col].replace({1.0: 1, 0.0: 0}).astype(int)
        df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0}).astype(int)
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )
        encoder.fit(df[self.one_hot_encode_columns])
        encoded_df = encoder.transform(df[self.one_hot_encode_columns])
        df = df.drop(columns=self.one_hot_encode_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Separate the classes
    df_y0 = df[df["Exited"] == 0].copy()
    df_y1 = df[df["Exited"] == 1].copy()

    # Find the smaller class size
    min_size = len(df_y1)

    # Randomly sample from each class
    df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

    # Concatenate back together
    df_balanced = pd.concat([df_y0_balanced, df_y1])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced
