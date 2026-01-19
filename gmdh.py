import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
#                       GMDH NEURON
# ============================================================

@dataclass
class Neuron:
    i: int
    j: int
    coef: Optional[np.ndarray] = None
    val_rmse: Optional[float] = None

    @staticmethod
    def design_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones_like(x1),
            x1,
            x2,
            x1 ** 2,
            x2 ** 2,
            x1 * x2
        ])

    def fit(self, x1_train: np.ndarray, x2_train: np.ndarray, y_train: np.ndarray) -> None:
        X = self.design_matrix(x1_train, x2_train)
        self.coef, *_ = np.linalg.lstsq(X, y_train, rcond=None)

    def predict(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        X = self.design_matrix(x1, x2)
        return X @ self.coef

    def score_val_rmse(self, x1_val: np.ndarray, x2_val: np.ndarray, y_val: np.ndarray) -> float:
        y_hat = self.predict(x1_val, x2_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_hat)))
        self.val_rmse = rmse
        return rmse



class GMDH:

    def __init__(self, max_layers=6, keep_k=10, patience=1):
        self.max_layers = max_layers
        self.keep_k = keep_k
        self.patience = patience

        self.layers_: List[List[Neuron]] = []
        self.best_neuron_: Optional[Neuron] = None
        self.best_layer_index_: Optional[int] = None
        self.history: List[Dict] = []

    def fit(self, X_train, y_train, X_val, y_val):

        train_signals = X_train.copy()
        val_signals = X_val.copy()

        best_rmse_overall = np.inf
        no_improve_count = 0

        for layer_idx in range(1, self.max_layers + 1):

            n_signals = train_signals.shape[1]
            if n_signals < 2:
                break

            candidates = []
            for i, j in combinations(range(n_signals), 2):
                neuron = Neuron(i=i, j=j)
                neuron.fit(train_signals[:, i], train_signals[:, j], y_train)
                neuron.score_val_rmse(val_signals[:, i], val_signals[:, j], y_val)
                candidates.append(neuron)

            candidates.sort(key=lambda n: n.val_rmse)
            selected = candidates[:min(self.keep_k, len(candidates))]
            self.layers_.append(selected)

            layer_best_rmse = selected[0].val_rmse
            layer_mean_rmse = float(np.mean([n.val_rmse for n in selected]))

            self.history.append({
                "layer": layer_idx,
                "num_candidates": len(candidates),
                "num_selected": len(selected),
                "best_val_rmse": layer_best_rmse,
                "mean_val_rmse_selected": layer_mean_rmse
            })

            if layer_best_rmse < best_rmse_overall:
                best_rmse_overall = layer_best_rmse
                self.best_neuron_ = selected[0]
                self.best_layer_index_ = layer_idx
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > self.patience:
                break

            next_train = []
            next_val = []
            for neuron in selected:
                next_train.append(neuron.predict(train_signals[:, neuron.i], train_signals[:, neuron.j]))
                next_val.append(neuron.predict(val_signals[:, neuron.i], val_signals[:, neuron.j]))

            train_signals = np.column_stack(next_train)
            val_signals = np.column_stack(next_val)

        return self

    def predict(self, X):
        signals = X.copy()

        for layer_idx in range(1, self.best_layer_index_ + 1):
            selected = self.layers_[layer_idx - 1]
            layer_outputs = [n.predict(signals[:, n.i], signals[:, n.j]) for n in selected]

            if layer_idx == self.best_layer_index_:
                return layer_outputs[0]

            signals = np.column_stack(layer_outputs)

        raise RuntimeError("Unexpected state in predict")

    def best_equation(self):
        if self.best_neuron_ is None:
            return "No best neuron."
        a = self.best_neuron_.coef
        return (
            f"y = {a[0]:.6f} + {a[1]:.6f}*x1 + {a[2]:.6f}*x2 + "
            f"{a[3]:.6f}*x1^2 + {a[4]:.6f}*x2^2 + {a[5]:.6f}*x1*x2"
        )




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
    DELTA_H = 6
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

    gmdh = GMDH(max_layers=15, keep_k=10, patience=1)
    gmdh.fit(X_train_s, y_train, X_val_s, y_val)

    y_pred = gmdh.predict(X_val_s)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = r2_score(y_val, y_pred)

    print("\nGMDH Results")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.4f} ({r2*100:.2f}%)")

    print("\nLayer history")
    for h in gmdh.history:
        print(h)

    print("\nBest equation")
    print(gmdh.best_equation())
