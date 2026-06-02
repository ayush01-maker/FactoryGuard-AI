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

    windows = [1, 6, 12]

    for col in sensor_cols:

        # Rolling Mean
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = (
                df[col]
                .rolling(window=w, min_periods=1)
                .mean()
            )

        # Rolling Standard Deviation
        for w in windows:
            df[f"{col}_roll_std_{w}"] = (
                df[col]
                .rolling(window=w, min_periods=1)
                .std()
            )

        # Exponential Moving Average
        df[f"{col}_ema"] = (
            df[col]
            .ewm(span=6, adjust=False)
            .mean()
        )

        # Lag Features
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_2"] = df[col].shift(2)

    # Handle NaN values
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

    # Check remaining NaNs
    print("Remaining NaNs:", df.isna().sum().sum())

    joblib.dump(
        df,
        "models/engineered_features.pkl"
    )

    print(
        "Engineered dataset saved to "
        "models/engineered_features.pkl"
    )