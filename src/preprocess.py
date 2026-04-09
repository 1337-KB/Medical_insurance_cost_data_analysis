import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df[["sex", "smoker", "region"]] = df[["sex", "smoker", "region"]].astype("category")
    return df


def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    df_reg = df.copy()
    df_reg["smoker_bin"] = (df_reg["smoker"] == "yes").astype(int)
    df_reg = df_reg.drop("smoker", axis=1)
    df_reg = pd.get_dummies(df_reg, columns=["sex", "region"], drop_first=True, dtype=int)
    return df_reg


def build_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    df_cls = df.copy()
    df_cls["smoker_bin"] = (df_cls["smoker"] == "yes").astype(int)
    df_cls = pd.get_dummies(df_cls, columns=["sex", "region"], drop_first=True, dtype=int)
    df_cls = df_cls.drop("smoker", axis=1)
    return df_cls