import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from joblib import Parallel, delayed

from util.dml import xgboost_model, repeat_dml

import warnings

warnings.simplefilter("ignore", UserWarning)

X_MARGINS = {
    # Binary
    "any_fatality": {"type": "binary", "params": {"p": 0.01}},
    "any_injury": {"type": "binary", "params": {"p": 0.05}},
    "aspiration_stay": {"type": "binary", "params": {"p": 0.9}},
    "place_identified": {"type": "binary", "params": {"p": 0.8}},
    "place_dependent": {"type": "binary", "params": {"p": 0.8}},
    "hometown": {"type": "binary", "params": {"p": 0.6}},
    "bin_dwell_tenure": {"type": "binary", "params": {"p": 0.1}},
    # Multinomial (mutually exclusive; hot-encoded)
    "structure": {
        "type": "multinomial",
        "columns": [
            "structure-concrete_heavy",
            "structure-wood_light",
            "structure-wood_heavy",
        ],
        "params": {"p": [0.7, 0.15, 0.15]},
    },
    "hazard_type": {
        "type": "multinomial",
        "columns": [
            "hazard_type-tsunami",
            "hazard_type-liquefaction",
            "hazard_type-earthquake_only",
        ],
        "params": {"p": [0.1, 0.1, 0.8]},
    },
    "occupation": {
        "type": "multinomial",
        "columns": [
            "occupation-agricultural",
            "occupation-business",
            "occupation-employment",
        ],
        "params": {"p": [0.3, 0.3, 0.4]},
    },
    # Ordinal
    # "home_damage": {
    #     "type": "ordinal",
    #     "params": {"levels": 4, "p": [0.35, 0.3, 0.3, 0.05]},
    # }, NOTE: major damage and completely destroyed are now grouped
    "home_damage": {
        "type": "ordinal",
        "params": {"levels": 3, "p": [0.35, 0.3, 0.35]},
    },
    "comm_damage": {
        "type": "ordinal",
        "params": {"levels": 4, "p": [0.2, 0.3, 0.3, 0.2]},
    },
    "bin_household_size": {
        "type": "ordinal",
        "params": {"levels": 5, "p": [0.2, 0.2, 0.2, 0.2, 0.2]},
    },
    "housing_quality": {
        "type": "ordinal",
        "params": {"levels": 5, "p": [0.05, 0.05, 0.2, 0.35, 0.35]},
    },
    "bin_income": {"type": "ordinal", "params": {"levels": 3, "p": [0.1, 0.7, 0.2]}},
    "edu_household": {
        "type": "ordinal",
        "params": {"levels": 4, "p": [0.05, 0.2, 0.25, 0.5]},
    },
    "bin_land_tenure": {
        "type": "ordinal",
        "params": {"levels": 4, "p": [0.7, 0.2, 0.05, 0.05]},
    },
}


def fit_dgp_nuisance_function(X, y):
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "n_jobs": 1,
        "eval_metric": "logloss",
        "scale_pos_weight": 1,
    }
    return XGBClassifier(**params).fit(X, y)


def simulate_dgp_margins(X_obs, treatment_model, outcome_model, theta, N=244, seed=333):
    """
    Generate synthetic dataset with true ATE = θ by independently sampling covariates X based ono target margins.

    Args:
        X_obs (pd.DataFrame): Observed covariates X
        theta (float): Assumed risk difference
        N (int): Assumed sample size
        seed (int): Random seed

    Returns:
        T_sim (pd.Series): Simulated binary treatment
        Y_sim (pd.Series): Simulated binary outcome
        X_sim (pd.DataFrame): Bootstrapped covariates
        true_ate (float): True ATE for this dataset (risk difference scale)
    """

    local_rng = np.random.default_rng(seed)

    # Generate covariates using target margins
    X_sim = pd.DataFrame(index=range(N))
    for cov_name, info in X_MARGINS.items():
        cov_type = info["type"]
        params = info["params"]
        if cov_type == "binary":
            p = params["p"]
            X_sim[cov_name] = local_rng.binomial(1, p, N)
        elif cov_type == "ordinal":
            levels = params["levels"]
            p = params["p"]
            X_sim[cov_name] = local_rng.choice(range(levels), size=N, p=p)
        elif cov_type == "multinomial":
            draws = local_rng.choice(len(params["p"]), size=N, p=params["p"])
            for k, cov_name in enumerate(info["columns"]):
                X_sim[cov_name] = (draws == k).astype(int)
        else:
            raise ValueError(f"Unsupported covariate type: {cov_type}")
    X_sim = X_sim[X_obs.columns]

    # Clip extreme scores
    pt = np.clip(treatment_model.predict_proba(X_sim)[:, 1], 0.0, 1.0)
    T_sim = pd.Series(local_rng.binomial(1, pt).astype(float))

    # Potential outcome probabilities under control (T=0) and treatment (T=1).
    py_base = outcome_model.predict_proba(X_sim)[:, 1]
    py_control = np.clip(py_base, 0.0, 1.0)
    py_treated = np.clip(py_base + theta, 0.0, 1.0)

    # Observed outcome follows treatment assignment
    py_observed = np.where(T_sim.values == 1, py_treated, py_control)
    Y_sim = pd.Series(local_rng.binomial(1, py_observed).astype(float))

    # True ATE: mean difference between potential outcomes
    true_ate = float(np.mean(py_treated - py_control))

    return T_sim, Y_sim, X_sim, true_ate


def simulate_dgp_observed(
    X_obs, treatment_model, outcome_model, theta, N=244, seed=111
):
    """
    Generate synthetic dataset with true ATE = θ by bootstapping observed data.

    Args:
        X_obs (pd.DataFrame): Observed covariates X
        treatment_model (method): Nuisance function for T ~ X
        outcome_model (method): Nuisance function for Y ~ X
        theta (float): Assumed risk difference
        N (int): Assumed sample size
        seed (int): Random seed

    Returns:
        T_sim (pd.Series): Simulated binary treatment
        Y_sim (pd.Series): Simulated binary outcome
        X_sim (pd.DataFrame): Bootstrapped covariates
        true_ate (float): True ATE for this dataset (risk difference scale)
    """
    local_rng = np.random.default_rng(seed)

    # Bootstrap X from observed data
    n_obs = len(X_obs)
    idx = local_rng.choice(n_obs, size=N, replace=True)
    X_sim = X_obs.iloc[idx].reset_index(drop=True)

    # Clip extreme scores
    pt = np.clip(treatment_model.predict_proba(X_sim)[:, 1], 0.0, 1.0)
    T_sim = pd.Series(local_rng.binomial(1, pt).astype(float))

    # Potential outcome probabilities under control (T=0) and treatment (T=1).
    py_base = outcome_model.predict_proba(X_sim)[:, 1]
    py_control = np.clip(py_base, 0.0, 1.0)
    py_treated = np.clip(py_base + theta, 0.0, 1.0)

    # Observed outcome follows treatment assignment.
    py_observed = np.where(T_sim.values == 1, py_treated, py_control)
    Y_sim = pd.Series(local_rng.binomial(1, py_observed).astype(float))

    # True ATE: mean difference between potential outcomes
    true_ate = float(np.mean(py_treated - py_control))

    return T_sim, Y_sim, X_sim, true_ate


def run_simulation(
    T_obs, Y_obs, X_obs, theta, seed, S=100, K=5, dgp=simulate_dgp_observed
):
    """
    Run one simulation using a DGP for a given effect size theta.

    Args:
        T (pd.Series): Observed treatment
        Y (pd.Series): Observed outcome
        X (pd.DataFrame): Observed covariates
        theta (float): Assumed risk difference
        seed (int): Random seed

    Returns:
        sig95 (int): 1 (pass) if CI excludes zero, 0 (fail) otherwise.
        theta_med (float): Median DML estimate across S repeats.
        true_ate (float): True ATE encoded in the DGP.
    """

    # Fit nuisance functions to observed data
    treatment_model = fit_dgp_nuisance_function(X_obs, T_obs)
    outcome_model = fit_dgp_nuisance_function(X_obs, Y_obs)

    # Data generating process
    T_sim, Y_sim, X_sim, true_ate = dgp(
        X_obs, treatment_model, outcome_model, theta, seed=int(seed)
    )

    # Replicate DML analysis; n_jobs=1 to avoid nested parallelization
    rds = repeat_dml(
        T_sim,
        Y_sim,
        X_sim,
        S=S,
        K=K,
        treatment_model=xgboost_model,
        outcome_model=xgboost_model,
        n_jobs=1,
        random_state=int(seed),
    )["rd_samples"]

    # Check desired confidence interval
    rd_ci95 = np.percentile(rds, [2.5, 97.5])
    sig95 = not (rd_ci95[0] <= 0 <= rd_ci95[1])
    rd_med = float(np.median(rds))

    return sig95, rd_med, true_ate


def simulated_power_curve(
    T_obs,
    Y_obs,
    X_obs,
    theta_grid,
    S_outer,
    target=0.8,
    power_seed=999,
    dgp=simulate_dgp_observed,
    n_jobs=-1,
    verbose=False,
):
    """Repeated power curve simulation for a provided range of effect sizes

    Args:
        T_obs (pd.Series): Observed treatment
        Y_obs (pd.Series): Observed outcome
        X_obs (pd.Series): Observed covariates
        theta_grid (list): Specified effect sizes
        S_outer (int): Number of power curve simulations
        target (float, optional): Target power. Defaults to 0.8.
        power_seed (int, optional): Random seed for power simulations. Defaults to 999.
        dgp (DGP method, optional): Data generating process function. Defaults to simulate_dgp_observed.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbose (bool, optional): Toggle print statements. Defaults to False.

    Returns:
        powers: Estimated power
        mean_ates: Mean (true) average treatment effects per DGP
        mean_thetas: Mean estimated effect sizes consistent with DML
    """
    rng = np.random.default_rng(power_seed)
    rep_seeds = rng.integers(0, 1_000_000, size=S_outer)

    if verbose:
        print(f"using DGP = {dgp}\n")
        print(f"{'θ (input)':>10}  {'true ATE':>10}  {'power':>8}  {'mean θ̂':>10}")
        print(
            "─--------------------------------------------------------------------------"
        )

    powers = []
    mean_ates = []
    mean_thetas = []

    # For each assumed effect
    for theta in theta_grid:

        # Monte Carlo power simulations
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_simulation)(T_obs, Y_obs, X_obs, theta, seed, dgp=dgp)
            for seed in rep_seeds
        )

        significant, theta_meds, true_ates = zip(*results)

        # Get averages across power simulations
        power = float(np.mean(significant))
        mean_ate = float(np.mean(true_ates))
        mean_theta = float(np.mean(theta_meds))

        # Append results
        powers.append(power)
        mean_ates.append(mean_ate)
        mean_thetas.append(mean_theta)
        if verbose:
            flag = "✓" if power >= target else " "
            print(
                f"{theta:>10.2f}  {mean_ate:>10.4f}  {power:>8.3f}  {mean_theta:>10.4f}  {flag}"
            )

    return powers, mean_ates, mean_thetas


def interpolate_mde(mean_ates, powers, target=0.8, verbose=False):
    """Linear interpolation to identify minimum detectable effect at a target power

    Args:
        mean_ates (pd.Series): Mean effect sizes
        powers (pd.Series): Simulated power
        target (float, optional): Desired power. Defaults to 0.8.
        verbose (bool, optional): Toggle print statements. Defaults to False.

    """
    mde = None
    for i in range(len(mean_ates) - 1):
        if powers[i] < target <= powers[i + 1]:
            x0, x1 = mean_ates[i], mean_ates[i + 1]
            y0, y1 = powers[i], powers[i + 1]
            mde = x0 + (target - y0) / (y1 - y0) * (x1 - x0)
            break

    if mde is None and len(powers) > 0 and powers[0] >= target:
        mde = mean_ates[0]

    if verbose:
        mde_str = (
            f"<= {mean_ates[0]:.3f}"
            if mde == mean_ates[0]
            else f"{mde:.3f}" if mde is not None else "not in effect grid"
        )
        print(f"MDE at {target:.0%} power : {mde_str}\n")
    return mde


# Plot power curve results if provided effect size and powers
def plot_power_curve(
    mean_ates,
    powers,
    ax,
    label=None,
    annotations=False,
    target=0.8,
    color="C0",
    linestyle="solid",
):
    ax.plot(
        mean_ates,
        powers,
        linewidth=1.5,
        color=color,
        marker="o",
        mfc="w",
        markeredgewidth=1.5,
        label=label,
        linestyle=linestyle,
        zorder=3,
    )
    if target:
        ax.axhline(target, color="dimgray", linewidth=0.8, linestyle="--", zorder=2)
    if annotations:
        for x, y in zip(mean_ates, powers):
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
            )
    ax.set(
        ylim=(0, 1.05),
        xlabel="Average effect",
        ylabel="Power",
    )
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.xaxis.grid(True)
    ax.legend(bbox_to_anchor=(1, 0.9))
    return ax
