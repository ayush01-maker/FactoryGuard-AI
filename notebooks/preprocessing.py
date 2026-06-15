import pandas as pd

def load_data(path):
    """
    Load and preprocess AI4I dataset.
    """

    df = pd.read_csv(path)

    # Remove unnecessary identifiers
    df.drop(
        columns=["UDI", "Product ID"],
        errors="ignore",
        inplace=True
    )

    return df


if __name__ == "__main__":
    df = load_data("Dataset/ai4i2020.csv")

    print("Dataset Shape:", df.shape)
    print(df.head())