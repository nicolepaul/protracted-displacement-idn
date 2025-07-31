import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


# Models for nuisance functions
def xgboost_model(is_binary, y=None):
    if is_binary:
        pos = y.sum()
        neg = len(y) - pos
        weight = neg / pos if pos > 0 else 1
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "n_jobs": 1,
            "eval_metric": "logloss",
            "scale_pos_weight": weight,
        }
        return XGBClassifier(**params)
    else:
        params = {"n_estimators": 100, "max_depth": 4, "n_jobs": 1}
        return XGBRegressor(**params)


# Single DML iteration
def dml(df, treatment, outcome, covariates, K=5, treatment_model=xgboost_model, outcome_model=xgboost_model, random_state=42):

    np.random.seed(random_state)

    X = df[covariates]
    T = df[treatment]
    Y = df[outcome]

    # Ensure binary T, Y
    is_binary_t = T.nunique() == 2
    is_binary_y = Y.nunique() == 2
    if (not is_binary_t) | (not is_binary_y):
        raise ValueError(
            f"This code is set up for binary T, Y only; T:{is_binary_t}, Y:{is_binary_y}"
        )

    # Ensure splits are stratified
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)

    # Initialize variables
    T_resid_all = np.zeros(len(df))
    Y_resid_all = np.zeros(len(df))
    T_hat_all = np.zeros(len(df))
    Y_hat_all = np.zeros(len(df))

    # Cross-fitting
    for train_idx, test_idx in kf.split(X, T):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        T_model = treatment_model(is_binary_t, y=T_train)
        T_model.fit(X_train, T_train)
        T_pred = (
            T_model.predict_proba(X_test)[:, 1]
            if is_binary_t
            else T_model.predict(X_test)
        )

        Y_model = outcome_model(is_binary_y, y=Y_train)
        Y_model.fit(X_train, Y_train)
        Y_pred = (
            Y_model.predict_proba(X_test)[:, 1]
            if is_binary_y
            else Y_model.predict(X_test)
        )

        T_hat_all[test_idx] = T_pred
        T_resid_all[test_idx] = T_test - T_pred

        Y_hat_all[test_idx] = Y_pred
        Y_resid_all[test_idx] = Y_test - Y_pred

    # Estimation
    reg = LinearRegression()
    reg.fit(T_resid_all.reshape(-1, 1), Y_resid_all)
    theta = reg.coef_[0]

    # Store results
    evaluation = {
        "seed": random_state,
        "treatment": treatment,
        "outcome": outcome,
        "theta": theta,
        "eval_Y": (
            roc_auc_score(Y, Y_hat_all)
            if is_binary_y
            else mean_squared_error(Y, Y_hat_all)
        ),
        "eval_T": (
            roc_auc_score(T, T_hat_all)
            if is_binary_t
            else mean_squared_error(T, T_hat_all)
        ),
    }

    return theta, evaluation


# Parallel sample repeats for DML
def repeat_dml(df, treatments, outcomes, covariates, S=100, K=5, treatment_model=xgboost_model, outcome_model=xgboost_model, n_jobs=-1, random_state=42):

    np.random.seed(random_state)

    results = []

    for outcome in outcomes:
        for treatment in treatments:

            seeds = np.random.randint(0, 1_000_000, size=S)
            dml_results = Parallel(n_jobs=n_jobs)(
                delayed(dml)(
                    df,
                    treatment,
                    outcome,
                    [c for c in covariates if c != treatment],
                    K=K,
                    treatment_model=treatment_model,
                    outcome_model=outcome_model,
                    random_state=seed,
                )
                for seed in seeds
            )

            thetas = [res[0] for res in dml_results]
            evaluations = [res[1] for res in dml_results]
            theta_median = np.median(thetas)

            results.append(
                {
                    "treatment": treatment,
                    "outcome": outcome,
                    "theta_median": theta_median,
                    "theta_stdev": np.std(thetas),
                    "theta_cov": np.std(thetas) / theta_median,
                    "theta_samples": thetas,
                    "evals": evaluations,
                }
            )

    return pd.DataFrame(results)


# Parse evaluation results
def extract_evaluation_table(results):
    rows = []
    for i, res in results.iterrows():
        for eval_row in res["evals"]:
            rows.append(
                {
                    "treatment": res["treatment"].split("-", 1).pop(),
                    "outcome": res["outcome"],
                    "theta": eval_row["theta"],
                    "eval_Y": eval_row["eval_Y"],
                    "eval_T": eval_row["eval_T"],
                    "seed": eval_row["seed"],
                }
            )
    return pd.DataFrame(rows)


# Plot causal effect results
def plot_results(ax, results_df, treatments, outcomes, data, var_label=None, colors=None, y_lims=(-0.5, 0.5), legend_title=None):
    x_offsets = np.linspace(-0.3, 0.3, num=len(outcomes))
    xtick_labels = [
        f'{t.split("-", 1).pop().split(") ").pop().replace(" damage"," ").replace("_", " ").capitalize()}\n(n={data[t].sum():,.0f})'
        for t in treatments
    ]

    for i, outcome in enumerate(outcomes):
        for j, treatment in enumerate(treatments):
            mask = (results_df["outcome"] == outcome) & (
                results_df["treatment"] == treatment
            )
            theta_samples = results_df.loc[mask, "theta_samples"].values
            if len(theta_samples) == 0:
                continue
            samples = theta_samples[0] if len(theta_samples) == 1 else theta_samples
            xpos = j + x_offsets[i]
            ax.boxplot(
                samples,
                positions=[xpos],
                widths=0.1,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i], color=colors[i]),
                medianprops=dict(color="black"),
                whiskerprops=dict(color=colors[i]),
                capprops=dict(color=colors[i]),
                flierprops=dict(markeredgecolor=colors[i], marker="o", markersize=3),
            )

    ax.axhline(0, linestyle="-", color="#212121", linewidth=0.5, zorder=2)
    ax.set_ylabel(var_label)
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(xtick_labels)
    ax.yaxis.grid(True)
    ax.set_ylim(y_lims)
    legend_handles = [
        Patch(
            color=colors[i],
            label=outcomes[i]
            .split("-")
            .pop()
            .replace("_", " ")
            .capitalize(),
        )
        for i in range(len(outcomes))
    ]
    ax.legend(handles=legend_handles, title=legend_title, bbox_to_anchor=(1, 0.9))


# Plot nuisance model evaluation
def plot_eval_metrics(eval_df, metric="eval_Y", title="Model performance", ylabel="AUC", colors=None):
    treatments = sorted(eval_df["treatment"].unique())
    outcomes = sorted(eval_df["outcome"].unique())

    x_offsets = np.linspace(-0.25, 0.25, num=len(treatments))

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, outcome in enumerate(outcomes):
        for j, treatment in enumerate(treatments):
            mask = (eval_df["outcome"] == outcome) & (eval_df["treatment"] == treatment)
            metric_vals = eval_df.loc[mask, metric].values
            xpos = i + x_offsets[j]

            ax.boxplot(
                metric_vals,
                positions=[xpos],
                widths=0.12,
                patch_artist=True,
                boxprops=dict(
                    facecolor=colors[j % len(colors)], color=colors[j % len(colors)]
                ),
                medianprops=dict(color="black"),
                whiskerprops=dict(color=colors[j % len(colors)]),
                capprops=dict(color=colors[j % len(colors)]),
                flierprops=dict(
                    markeredgecolor=colors[j % len(colors)], marker="o", markersize=3
                ),
            )

    ax.axhline(1.0, linestyle="-", color="#212121", linewidth=0.8)
    ax.axhline(0.5, linestyle="-", color="darkred", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(
        [
            o.split("-").pop().replace("_", " ").capitalize()
            for o in outcomes
        ],
        rotation=90,
    )
    ax.yaxis.grid(True)

    legend_handles = [
        Patch(color=colors[i % len(colors)], label=t) for i, t in enumerate(treatments)
    ]
    ax.legend(
        handles=legend_handles,
        title="Treatment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()
