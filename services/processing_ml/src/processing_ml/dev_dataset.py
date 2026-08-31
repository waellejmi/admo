import pandas as pd

from .preprocessing import load_data, preprocess


def create_dev_dataset(
    file_path: str, n_rows: int = 100000, seed: int = 42
) -> pd.DataFrame:
    df = load_data(file_path)
    df = preprocess(df)

    sample_size = min(n_rows, len(df))
    df = df.sample(n=sample_size, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    df = create_dev_dataset("data/OPERAT03_RATIO_CONSO_AJUSTEE.csv")
    df.to_csv("data/dev_dataset.csv", index=False)
