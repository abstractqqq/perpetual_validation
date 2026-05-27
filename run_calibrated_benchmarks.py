import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from perpetual import PerpetualBooster
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
N_TRIALS = 100


class SimpleCalibrator:
    """Manual Sigmoid (Platt) Calibration using Cross-Validation"""

    def __init__(self, base_model_type, params=None, cv=3):
        self.base_model_type = base_model_type
        self.params = params or {}
        self.cv = cv
        self.calibrators = []

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y)
        kf = KFold(n_splits=self.cv, shuffle=False)

        self.calibrators = []
        for train_idx, val_idx in kf.split(X_arr):
            X_t, X_v = X_arr[train_idx], X_arr[val_idx]
            y_t, y_v = y_arr[train_idx], y_arr[val_idx]

            if self.base_model_type == "perpetual":
                model = PerpetualBooster(objective="LogLoss")
                model.fit(X_t, y_t)
            else:
                dtrain = lgb.Dataset(X_t, label=y_t)
                model = lgb.train(self.params, dtrain)

            if self.base_model_type == "perpetual":
                raw_probs = model.predict_proba(X_v)[:, 1]
            else:
                raw_probs = model.predict(X_v)

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty=None)
            lr.fit(raw_probs.reshape(-1, 1), y_v)

            self.calibrators.append((model, lr))
        return self

    def predict_proba(self, X):
        X_arr = np.array(X)
        all_probs = []
        for model, lr in self.calibrators:
            if self.base_model_type == "perpetual":
                raw = model.predict_proba(X_arr)[:, 1]
            else:
                raw = model.predict(X_arr)
            calibrated = lr.predict_proba(raw.reshape(-1, 1))[:, 1]
            all_probs.append(calibrated)

        mean_prob = np.mean(all_probs, axis=0)
        return np.vstack([1 - mean_prob, mean_prob]).T


def evaluate(model, X, y, split_name):
    probs = model.predict_proba(X)[:, 1]
    return {
        f"{split_name} ROC AUC": roc_auc_score(y, probs),
        f"{split_name} LogLoss": log_loss(y, probs),
    }


def run_calibrated_benchmark(X, y, name, has_oos=False):
    print(f"\n--- Running Calibrated Benchmark (Manual): {name} ---")

    if has_oos:
        n = len(X)
        train_idx = int(n * 0.6)
        test_idx = int(n * 0.8)
        X_train, y_train = X.iloc[:train_idx], y[:train_idx]
        X_test, y_test = X.iloc[train_idx:test_idx], y[train_idx:test_idx]
        X_oos, y_oos = X.iloc[test_idx:], y[test_idx:]
    else:
        X_train_full, X_oos, y_train_full, y_oos = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42
        )

    results = {}

    print("Training Calibrated Perpetual...")
    start_p = time.time()
    cal_p = SimpleCalibrator("perpetual", cv=3)
    cal_p.fit(X_train, y_train)
    p_time = time.time() - start_p

    res_p = evaluate(cal_p, X_train, y_train, "Train")
    res_p.update(evaluate(cal_p, X_test, y_test, "Test"))
    res_p.update(evaluate(cal_p, X_oos, y_oos, "OOS"))
    res_p["Time (s)"] = p_time
    results["Calibrated Perpetual"] = res_p

    print(f"Tuning LightGBM with Optuna ({N_TRIALS} trials)...")

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        gbm = lgb.train(
            param,
            lgb.Dataset(X_t, label=y_t),
            valid_sets=[lgb.Dataset(X_v, label=y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return log_loss(y_v, gbm.predict(X_v))

    start_optuna = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_params
    best.update({"objective": "binary", "verbosity": -1})

    print("Training Calibrated LightGBM...")
    cal_l = SimpleCalibrator("lgbm", params=best, cv=3)
    cal_l.fit(X_train, y_train)
    l_time = time.time() - start_optuna

    res_l = evaluate(cal_l, X_train, y_train, "Train")
    res_l.update(evaluate(cal_l, X_test, y_test, "Test"))
    res_l.update(evaluate(cal_l, X_oos, y_oos, "OOS"))
    res_l["Time (s)"] = l_time
    results["Optuna + Calibrated LGBM"] = res_l

    df = pd.DataFrame(results).T
    print(df)
    return df


def main():
    all_res = {}

    print("\nLoading Electricity dataset...")
    data_elec = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df_elec = (
        data_elec.frame.copy().sort_values(by=["date", "period"]).reset_index(drop=True)
    )
    X_el = df_elec.drop(columns=["class"])
    y_el = LabelEncoder().fit_transform(df_elec["class"].astype(str))
    all_res["Electricity"] = run_calibrated_benchmark(
        X_el, y_el, "Electricity", has_oos=True
    )

    print("\nLoading Bank dataset...")
    data_bank = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    X_bk = data_bank.data.copy()
    for col in X_bk.select_dtypes(["category", "object"]).columns:
        X_bk[col] = LabelEncoder().fit_transform(X_bk[col].astype(str))
    y_bk = LabelEncoder().fit_transform(data_bank.target.astype(str))
    all_res["Bank"] = run_calibrated_benchmark(
        X_bk, y_bk, "Bank Marketing", has_oos=False
    )

    summary_list = []
    for ds, df in all_res.items():
        for model in df.index:
            row = {"Dataset": ds, "Model": model}
            row.update(df.loc[model].to_dict())
            summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv("calibrated_benchmark_summary.csv", index=False)
    print("\n--- FINAL CALIBRATED SUMMARY ---")
    print(df_summary)

    for metric in ["ROC AUC", "LogLoss"]:
        plt.figure(figsize=(14, 7))
        cols = [f"Train {metric}", f"Test {metric}", f"OOS {metric}"]
        df_melt = df_summary.melt(
            id_vars=["Dataset", "Model"],
            value_vars=cols,
            var_name="Split",
            value_name="Value",
        )
        sns.barplot(data=df_melt, x="Dataset", y="Value", hue="Model", palette="muted")
        plt.title(f"{metric} Comparison: Train vs Test vs OOS (Manual Calibration)")
        plt.savefig(f"calibrated_{metric.lower().replace(' ', '_')}_comparison.png")

    # Time vs ROC AUC Visualization
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(
        data=df_summary,
        x="Time (s)",
        y="OOS ROC AUC",
        hue="Model",
        style="Dataset",
        s=200,
        palette="Set1",
    )
    plt.xscale("log")
    plt.title("Efficiency Frontier: Training Time (Log Scale) vs. OOS ROC AUC")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    for i in range(df_summary.shape[0]):
        plt.text(
            df_summary["Time (s)"][i] * 1.1,
            df_summary["OOS ROC AUC"][i],
            f"{df_summary['Dataset'][i]} ({df_summary['Model'][i].split()[-1]})",
            fontsize=9,
            alpha=0.7,
        )

    plt.savefig("calibrated_time_vs_auc.png")
    print("Efficiency plot saved as calibrated_time_vs_auc.png")


if __name__ == "__main__":
    main()
