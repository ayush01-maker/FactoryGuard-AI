import pandas as pd
import joblib


def create_features(df):

    sensor_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    for col in sensor_cols:

        # ----------------------------------
        # Rolling Mean (1h, 6h, 12h)
        # ----------------------------------

        df[f"{col}_mean_1h"] = (
            df[col]
            .rolling(window=1, min_periods=1)
            .mean()
        )

        df[f"{col}_mean_6h"] = (
            df[col]
            .rolling(window=6, min_periods=1)
            .mean()
        )

        df[f"{col}_mean_12h"] = (
            df[col]
            .rolling(window=12, min_periods=1)
            .mean()
        )

        # ----------------------------------
        # Standard Deviation (1h, 6h, 12h)
        # ----------------------------------

        df[f"{col}_std_1h"] = (
            df[col]
            .rolling(window=1, min_periods=1)
            .std()
        )

        df[f"{col}_std_6h"] = (
            df[col]
            .rolling(window=6, min_periods=1)
            .std()
        )

        df[f"{col}_std_12h"] = (
            df[col]
            .rolling(window=12, min_periods=1)
            .std()
        )

        # ----------------------------------
        # Exponential Moving Average
        # ----------------------------------

        df[f"{col}_ema"] = (
            df[col]
            .ewm(span=6, adjust=False)
            .mean()
        )

        # ----------------------------------
        # Lag Features
        # ----------------------------------

        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_2"] = df[col].shift(2)

    # ----------------------------------
    # Handle Missing Values
    # ----------------------------------

    df = df.bfill()
    df = df.ffill()
    df = df.fillna(0)

    return df


if __name__ == "__main__":

    df = pd.read_csv("Dataset/ai4i2020.csv")

    df.drop(
        columns=["UDI", "Product ID"],
        errors="ignore",
        inplace=True
    )

    df = create_features(df)

    print("Feature Engineering Completed")
    print("Shape:", df.shape)

    print("Remaining NaNs:", df.isna().sum().sum())

    joblib.dump(
        df,
        "models/engineered_features.pkl"
    )

    print(
        "Engineered dataset saved to "
        "models/engineered_features.pkl"
    )
