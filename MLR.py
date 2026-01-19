import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def make_time_key(df):
    return pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")


def build_forecast_df(df, features_now, target_col, delta_h=1, add_lags=True):
    df = df.copy()

    df = df.drop_duplicates(subset=["date", "segmentId", "hour"])
    df = df.dropna(subset=features_now + [target_col])

    df["time_key"] = make_time_key(df)
    df = df.sort_values(["segmentId", "time_key"])

    future_target = f"{target_col}_t_plus_{delta_h}h"
    df[future_target] = df.groupby("segmentId")[target_col].shift(-delta_h)

    final_features = list(features_now)

    if add_lags:
        for lag in [1, 2, 3]:
            df[f"avgSpeed_lag{lag}h"] = df.groupby("segmentId")[target_col].shift(lag)
            df[f"travelTimeRatio_lag{lag}h"] = df.groupby("segmentId")["travelTimeRatio"].shift(lag)
            final_features += [
                f"avgSpeed_lag{lag}h",
                f"travelTimeRatio_lag{lag}h"
            ]

    df = df.dropna(subset=final_features + [future_target])
    return df, final_features, future_target


def temporal_split(df, train_frac=0.75):
    df = df.sort_values("time_key")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


if __name__ == "__main__":
    INPUT_CSV = "flattened_traffic_data.csv"
    DELTA_H = 1
    TRAIN_FRAC = 0.75

    FEATURES_NOW = [
        "speedLimit",
        "distance",
        "frc",
        "hour",
        "travelTimeRatio",
        "standardDeviationSpeed",
        "sampleSize"
    ]
    TARGET_COL = "averageSpeed"

    df_raw = pd.read_csv(INPUT_CSV)

    df_forecast, FEATURES, TARGET = build_forecast_df(
        df_raw,
        features_now=FEATURES_NOW,
        target_col=TARGET_COL,
        delta_h=DELTA_H,
        add_lags=True
    )

    df_train, df_val = temporal_split(df_forecast, train_frac=TRAIN_FRAC)

    X_train = df_train[FEATURES].astype(float).values
    y_train = df_train[TARGET].astype(float).values

    X_val = df_val[FEATURES].astype(float).values
    y_val = df_val[TARGET].astype(float).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    mlr = LinearRegression()
    mlr.fit(X_train_s, y_train)

    y_pred = mlr.predict(X_val_s)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = r2_score(y_val, y_pred)

    print("\nMLR Results")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.4f} ({r2*100:.2f}%)")

    print("\nCoefficients:")
    for name, coef in zip(FEATURES, mlr.coef_):
        print(f"{name}: {coef:.6f}")
    print(f"Intercept: {mlr.intercept_:.6f}")
