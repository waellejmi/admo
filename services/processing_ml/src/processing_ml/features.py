import pandas as pd


def add_occupation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_mono_occupation"] = (df["cas_assujettissement_efa"] == "1A").astype(int)
    return df
