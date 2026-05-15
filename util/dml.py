import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from util.evalues import rd_to_rr, e_value_point, e_value_ci


# Models for nuisance functions - per Fuhr et al. (2024) I am using XGBoost
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
def dml(
    T,
    Y,
    X,
    K=5,
    treatment_model=xgboost_model,
    outcome_model=xgboost_model,
    random_state=42,
):
    """Perform a single iteration of DML

    Args:
        T (pd.Series): Treatment (or exposure) for each sample. Assumed binary
        Y (pd.Series): Outcome for each sample. Assumed binary.
        X (pd.DataFrame): Covariates to be controlled for.
        K (int, optional): Number of cross-validation folds in DML. Defaults to 5.
        treatment_model (sklearn-compatible classifier function, optional): Type of classifier to be used for the treatment model T ~ X. Defaults to xgboost_model.
        outcome_model (sklearn-compatible classifier function, optional): Type of classifier to be used for the outcome model Y ~ X. Defaults to xgboost_model.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        theta_rd: Average treatment effect, risk difference
        theta_rr: Average treatment effect, relative risk
        evaluation (dict): Nuisance model diagnostics
        se (float): Heteroskedasticity-robust standard error of theta_rd, computed via the sandwich formula on the cross-fitted residuals
    """
    # Set seed
    np.random.seed(random_state)

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
    T_resid_all = np.zeros(len(T))
    Y_resid_all = np.zeros(len(T))
    T_hat_all = np.zeros(len(T))
    Y_hat_all = np.zeros(len(T))

    # Cross-fitting
    for train_idx, test_idx in kf.split(X, T):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        # Treatment model
        T_model = treatment_model(is_binary_t, y=T_train)
        T_model.fit(X_train, T_train)
        T_pred = (
            T_model.predict_proba(X_test)[:, 1]
            if is_binary_t
            else T_model.predict(X_test)
        )

        # Outcome model
        Y_model = outcome_model(is_binary_y, y=Y_train)
        Y_model.fit(X_train, Y_train)
        Y_pred = (
            Y_model.predict_proba(X_test)[:, 1]
            if is_binary_y
            else Y_model.predict(X_test)
        )

        # Treatment residuals
        T_hat_all[test_idx] = T_pred
        T_resid_all[test_idx] = T_test - T_pred

        # Outcome residuals
        Y_hat_all[test_idx] = Y_pred
        Y_resid_all[test_idx] = Y_test - Y_pred

    # Estimation using OLS
    reg = LinearRegression()
    reg.fit(T_resid_all.reshape(-1, 1), Y_resid_all)
    theta_rd = reg.coef_[0]

    # Sensitivity: E-values
    p0_hat = Y[T == 0].mean()  # baseline risk
    theta_rr = rd_to_rr(theta_rd, p0_hat)  # risk difference to relative risk

    # Store nuisance function evaluation results
    evaluation = {
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

    return theta_rd, theta_rr, evaluation


# Parallel sample repeats for DML
def repeat_dml(
    T,
    Y,
    X,
    S=100,
    K=5,
    treatment_model=xgboost_model,
    outcome_model=xgboost_model,
    n_jobs=-1,
    random_state=42,
):
    """Perform DML for the requeted number of S repeats in parallel.

    Args:
        T (pd.Series): Treatment (or exposure) for each sample. Assumed binary
        Y (pd.Series): Outcome for each sample. Assumed binary.
        X (pd.DataFrame): Covariates to be controlled for.
        S (int, optional): Number of repeats in DML. Defaults to 100.
        K (int, optional): Number of cross-validation folds in DML. Defaults to 5.
        treatment_model (sklearn-compatible classifier function, optional): Type of classifier to be used for the treatment model T ~ X. Defaults to xgboost_model.
        outcome_model (sklearn-compatible classifier function, optional): Type of classifier to be used for the outcome model Y ~ X. Defaults to xgboost_model.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: Average treatment effects and E-values for each repeat; nuisance model evaluation
    """
    np.random.seed(random_state)
    seeds = np.random.randint(0, 1_000_000, size=S)

    dml_results = Parallel(n_jobs=n_jobs)(
        delayed(dml)(
            T,
            Y,
            X,
            K=K,
            treatment_model=treatment_model,
            outcome_model=outcome_model,
            random_state=seed,
        )
        for seed in seeds
    )

    theta_rds = [res[0] for res in dml_results]
    theta_rrs = [res[1][0] for res in dml_results]
    e_values = [e_value_point(rr)[0] for rr in theta_rrs]
    evaluations = [res[2] for res in dml_results]

    return {
        "rd_samples": theta_rds,
        "rr_samples": theta_rrs,
        "e_value_samples": e_values,
        "evals": evaluations,
    }


# Summarize estimation results
def summarize_results(rds, rrs):
    """Summarize the average treatment effects in risk difference and relative risk scales

    Args:
        rds (list): Average treatment effects using a risk difference scale for each repeat
        rrs (list): Average treatment effects using a relative risk scale for each repeat

    Returns:
        dict: Median average treatment effects, confidence intervals, and E-values
    """
    rd_med = np.median(rds)
    rr_med = np.median(rrs)

    rd_ci95 = np.percentile(rds, [2.5, 97.5])
    rr_ci95 = np.percentile(rrs, [2.5, 97.5])
    sig95 = not (rd_ci95[0] <= 0 <= rd_ci95[1])

    e_point, e_point_flip = e_value_point(rr_med)
    e_ci, e_ci_flip = e_value_ci(rr_ci95[0], rr_ci95[1])

    return {
        "rd_median": rd_med,
        "rr_median": rr_med,
        "rd_ci95": rd_ci95,
        "rr_ci95": rr_ci95,
        "sig95": sig95,
        "e_point": e_point,
        "e_ci": e_ci,
        "e_point_flip": e_point_flip,
        "e_ci_flip": e_ci_flip,
    }


# Summarize nuisance model diagnostics
def summarize_nuisance(evals):
    eval_T = [e["eval_T"] for e in evals]
    eval_Y = [e["eval_Y"] for e in evals]

    return {
        "treatment_auc_mean": np.mean(eval_T),
        "treatment_auc_sd": np.std(eval_T),
        "outcome_auc_mean": np.mean(eval_Y),
        "outcome_auc_sd": np.std(eval_Y),
    }


# Main analysis function
def run_analysis(df, treatments, outcomes, covariates, S=100, K=5):
    """Main function to run DML analysis to estimate average treatment effects for a set of binary treatments and binary outcomes

    Args:
        df (pd.DataFrame): Observational data that the causal effect estimation is performed on
        treatments (list): List of strings indicating the columns in df to be used as binary treatments
        outcomes (list): List of strings indicating the columns in df to be used as binary outcomes
        covariates (list): List of strings indicating the columns in df to be controlled for
        S (int, optional): Number of repeats to be performed during DML. Defaults to 100.
        K (int, optional): Number of cross-validation folds to be used in DML. Defaults to 5.

    Returns:
        pd.DataFrame: Summary of the DML results for each set of treatment and outcome
    """
    rows = []

    for treatment in treatments:
        for outcome in outcomes:

            # Run analysis
            repeated = repeat_dml(
                df[treatment],
                df[outcome],
                df[[c for c in covariates if c != treatment]],
                S,
                K,
            )

            # Extract summaries
            effect = summarize_results(repeated["rd_samples"], repeated["rr_samples"])
            nuisance = summarize_nuisance(repeated["evals"])

            # Assemble results
            rows.append(
                {
                    "treatment": treatment,
                    "outcome": outcome,
                    **effect,
                    **nuisance,
                    "rd_samples": repeated["rd_samples"],
                    "rr_samples": repeated["rr_samples"],
                    "evals": repeated["evals"],
                }
            )

    return pd.DataFrame(rows)


# Parse evaluation results
def extract_evaluation_table(results):
    rows = []
    for i, res in results.iterrows():
        for eval_row in res["evals"]:
            rows.append(
                {
                    "treatment": res["treatment"].split("-", 1).pop(),
                    "outcome": res["outcome"],
                    "theta": res["rd_median"],
                    "eval_Y": eval_row["eval_Y"],
                    "eval_T": eval_row["eval_T"],
                }
            )
    return pd.DataFrame(rows)


# Plot causal effect results
def plot_results(
    ax,
    results_df,
    treatments,
    outcomes,
    data,
    var_label=None,
    colors=None,
    y_lims=(-0.55, 0.55),
    legend_title=None,
):
    x_offsets = np.linspace(-0.3, 0.3, num=len(outcomes))
    xtick_labels = [
        f'{t.split("-", 1).pop().split(") ").pop().replace(" damage"," ").replace("_", " ").replace('place ','place\n').replace('bove ', 'bove\n').replace('elow ', 'elow\n').capitalize()}\n(n={data[t].sum():,.0f})'
        for t in treatments
    ]

    for i, outcome in enumerate(outcomes):
        for j, treatment in enumerate(treatments):
            mask = (results_df["outcome"] == outcome) & (
                results_df["treatment"] == treatment
            )
            theta_samples = results_df.loc[mask, "rd_samples"].values
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
            label=outcomes[i].split("-").pop().replace("_", " ").capitalize(),
        )
        for i in range(len(outcomes))
    ]
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        bbox_to_anchor=(1, 0.9),
        handlelength=0.6,
        handleheight=0.6,
    )


# Plot nuisance model evaluation
def plot_eval_metrics(
    eval_df, metric="eval_Y", title="Model performance", ylabel="AUC", colors=None
):
    treatments = sorted(eval_df["treatment"].unique())
    outcomes = sorted(eval_df["outcome"].unique())

    x_offsets = np.linspace(-0.25, 0.25, num=len(treatments))

    fig, ax = plt.subplots(figsize=(8, 2.25))

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
        [o.split("-").pop().replace("_", " ").capitalize() for o in outcomes],
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
