import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec


def plot_config(theme=None):

    # Font
    FPT = 9
    FSIZE = "medium"
    FNAME = "Arial"

    # Figure size
    FIGSIZE = [6, 4]

    # Colors
    COLOR = "212121"
    colors = None
    if not theme:
        print("Using default theme")
        colors = matplotlib.cycler(  # type: ignore
            "color",
            [
                "44aa98",
                "ab4498",
                "332389",
                "86ccec",
                "ddcc76",
                "cd6477",
                "882255",
                "117732",
            ],
        )
    elif theme == "derisc":
        print(f"Using theme: {theme}")
        colors = matplotlib.cycler(
            "color", ["8f993e", "b4bd01", "555025", "bbc493", "efede7"]
        )
    elif theme == "derisc_pres":
        print(f"Using theme: {theme}")
        FPT = 10.5
        FIGSIZE = [4, 4]
        FNAME = "Montserrat"
        COLOR = "8d8379"
        # plt.rcParams['font.weight'] = 'light'
        colors = matplotlib.cycler(
            "color", ["8f993e", "b4bd01", "555025", "bbc493", "efede7"]
        )

    # Other styling
    plt.rcParams["font.size"] = FPT
    plt.rcParams["figure.figsize"] = FIGSIZE
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR
    plt.rcParams["font.family"] = FNAME
    plt.rcParams["axes.facecolor"] = "None"
    plt.rcParams["axes.edgecolor"] = "dimgray"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["grid.linestyle"] = "dashed"
    plt.rcParams["xtick.labelsize"] = FSIZE
    plt.rcParams["ytick.labelsize"] = FSIZE
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "None"
    plt.rcParams["legend.fontsize"] = FSIZE
    plt.rcParams["legend.title_fontsize"] = FSIZE
    plt.rcParams["axes.labelsize"] = FSIZE
    plt.rcParams["savefig.facecolor"] = "None"
    plt.rcParams["savefig.edgecolor"] = "None"
    plt.rcParams["axes.titlesize"] = FSIZE
    plt.rcParams["legend.handlelength"] = 1
    plt.rcParams["legend.handleheight"] = 1
    plt.rc("axes", prop_cycle=colors)


def get_luminance(r, g, b):
    return 255.0 * (r * 0.299 + g * 0.587 + b * 0.114)


def get_contrast_color(r, g, b):

    # Default color choices
    light_color, dark_color = "w", "#212121"
    # Luminance threshold
    L = 186

    # Determine font color based on contrast
    contrast_color = dark_color if get_luminance(r, g, b) > L else light_color

    # Return result
    return contrast_color


# Add labels with appropriate color contrast
def add_bar_labels(ax, fmt="percent", color=None, min_value=0.05):

    # Load configuration
    # plot_config()

    # Luminance threshold
    L = 186

    # Add labels for each container set
    for _, c in enumerate(ax.containers):
        # Get bar color
        r, g, b, a = c.patches[0].get_facecolor()
        # Determine font color based on contrast
        font_color = color
        if not color:
            font_color = get_contrast_color(r, g, b)
        labels = [f"{x:.0%}" if x > min_value else "" for x in c.datavalues]
        if fmt == "number":
            labels = [f"{x:,.0f}" if x > 1 else "" for x in c.datavalues]
        # Add labels
        ax.bar_label(
            c, labels=labels, label_type="center", fontsize="small", color=font_color
        )

    # Return result
    return ax


# Estimate correlations
def create_correlation_matx(data, corr_tol=0.7, absolute=True, method="spearman"):

    # Determine (absolute value of) correlations
    corr_matx = data.corr(method=method)
    if absolute:
        corr_matx = corr_matx.abs()

    # Get upper and lower triangle
    upper = corr_matx.where(np.triu(np.ones(corr_matx.shape), k=1).astype(bool))
    lower = corr_matx.where(np.tril(np.ones(corr_matx.shape)).astype(bool))

    # Find features with correlation greater than a given threshold
    to_drop = [
        column for column in upper.columns if any(upper[column].abs() > corr_tol)
    ]
    drop_str = "\n".join([f"    {drop}" for drop in to_drop])
    print(
        f"These {len(to_drop):,.0f} columns exceed the tolerance of {corr_tol:.0%}:\n{drop_str}\n"
    )

    # Return result
    return corr_matx, upper, lower, to_drop


# Plot correlation matrix
def plot_correlation_matx(corr_matx, ax, cmap=None, groups=None, **kwds):

    # Determine data limits
    vmin = -1 if corr_matx.min().min() < 0 else 0
    vmax = 1

    # Create colormap
    if not cmap:
        cmap = "bwr_r" if vmin == -1 else "Greens"
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color="silver", alpha=0.6)

    # Build friendly label mapping from groups
    label_map = {}
    if groups:
        for group_vars in groups.values():
            label_map.update(group_vars)
    friendly = [label_map.get(v, v) for v in corr_matx.index]

    # Plot gridded values
    ax.imshow(corr_matx, cmap=cmap, vmin=vmin, vmax=vmax, **kwds)

    # Apply friendly tick labels
    ax.set_xticks(range(len(friendly)), labels=friendly, rotation=90)
    ax.set_yticks(range(len(friendly)), labels=friendly)

    # Add data labels
    for (j, i), val in np.ndenumerate(corr_matx):
        if val == val:
            label = f"{val:.0%}"
            r, g, b, a = cmap((val - vmin) / (vmax - vmin))
            font_color = get_contrast_color(r, g, b)
            ax.text(i, j, label, ha="center", va="center", color=font_color)

    # Group dividers and labels
    if groups:
        group_sizes = [len(v) for v in groups.values()]
        group_names = list(groups.keys())
        boundaries  = np.cumsum(group_sizes)

        # Dividers between groups
        for b in boundaries[:-1]:
            for xy in [ax.axhline, ax.axvline]:
                xy(b - 0.5, color='white', linewidth=3)

        # Group labels
        start = 0
        for name, size in zip(group_names, group_sizes):
            mid = start + size / 2 - 0.5
            ax.annotate(
                name,
                xy=(mid, -0.5), xycoords='data',
                xytext=(0, 6), textcoords='offset points',
                ha='center', va='bottom',
                fontstyle='italic',
                annotation_clip=False,
            )
            start += size

    return ax


# Hard-coded inputs for ECDF plots
COLORS = {
    "duration_total": ["#d8d6cc", "#c1c78a", "#7b8230", "#3f4625"],
    "duration_emergency": ["#e7d3ba", "#f5ab5e", "#ea7600", "#733c1c"],
}

DAYS_IN_MONTH = 30.437


XTICKS = {
    "duration_emergency": np.array(
        [7 / DAYS_IN_MONTH, 14 / DAYS_IN_MONTH, 1, 3, 6, 12, 36]
    ),
    "duration_total": np.array([7 / DAYS_IN_MONTH, 1, 3, 6, 12, 36, 72]),
}

XTICKLABELS = {
    "duration_emergency": ["1w", "2w", "1m", "3m", "6m", "1y", "3y"],
    "duration_total": ["1w", "1m", "3m", "6m", "1y", "3y", "6y"],
}


# Empriical cumulative distribution function
def compute_ecdf(x):
    X = np.sort(x)
    Y = np.arange(1, len(X) + 1) / float(len(X))
    return X, Y


# Custom string handling to save space
def clean_label(raw, factor):
    label = str(raw)
    if "_damage" in factor:
        label = label.replace("ly lost", "").split(" ")[1]
    if "income" in factor:
        label = label.replace("minimum", "min")
    if "edu_" in factor:
        label = label.replace(" or ", "/")
    return label.capitalize()


# Use consistent color palettes
def palette4_to_n(palette, n):
    if n == 2:
        return [palette[1], palette[-1]]
    elif n == 3:
        return [palette[0], palette[2], palette[-1]]
    elif n == 4:
        return palette
    elif n == 5:
        return palette + ["#212121"]
    else:
        raise ValueError(
            f"Can only convert the discrete 4-color palette into 2 to 5 colors; requested {n}"
        )


# Group and plot stratified empirical cumulative distribution functions
def plot_stratified_ecdf(df, pf, category, outcome_dict, variables):

    # Parse dimensions
    outcomes = list(outcome_dict.keys())
    outcome_labels = list(outcome_dict.values())

    factors = list(variables.keys())
    displays = list(variables.values())

    n_rows = len(factors)
    n_cols = len(outcomes)

    # Initialize figure with a dedicated legend column
    SUBPLOT_W = 2.8
    SUBPLOT_H = 1.5
    LEGEND_W = 1.4
    fig = plt.figure(
        figsize=(n_cols * (SUBPLOT_W + LEGEND_W), n_rows * SUBPLOT_H),
        constrained_layout=True,
    )
    gs = gridspec.GridSpec(
        n_rows,
        n_cols * 2,
        figure=fig,
        width_ratios=[SUBPLOT_W, LEGEND_W] * n_cols,
    )
    fig.suptitle(category)

    # Construct plot
    for i, (factor, display) in enumerate(zip(factors, displays)):
        unique_vals = np.sort(pf[factor][pf[factor] >= 0].dropna().unique())

        for j, (outcome, outcome_label) in enumerate(zip(outcomes, outcome_labels)):

            ax = fig.add_subplot(gs[i, j * 2])

            palette = COLORS[outcome]
            colors = palette4_to_n(palette, len(unique_vals))
            handles = []

            for k, val in enumerate(unique_vals):
                idx = pf[factor] == val
                n = idx.sum()
                ecdf_x, ecdf_y = compute_ecdf(pf.loc[idx, outcome].dropna())

                if df[factor].dtype == bool:
                    raw_label = str(bool(val))
                else:
                    raw_label = df[factor].cat.categories[int(val)]
                label_str = clean_label(raw_label, factor)

                (line,) = ax.plot(
                    ecdf_x, ecdf_y, label=f"{label_str} (n={n:.0f})", color=colors[k]
                )
                handles.append(line)

            ax.set(
                xlabel=outcome_label,
                ylabel="Proportion",
                title=display,
                xscale="log",
            )
            ax.set_xticks(XTICKS[outcome])
            ax.set_xticklabels(XTICKLABELS[outcome])
            ax.set_xlim(0.1, XTICKS[outcome][-1])
            ax.yaxis.set_major_formatter("{x:.0%}")
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)

            legend_ax = fig.add_subplot(gs[i, j * 2 + 1])
            legend_ax.axis("off")
            legend_ax.legend(handles=handles, loc="center left", frameon=False)

    # fig.tight_layout()
    fig.savefig(
        os.path.join("img", f"pub_stratified_{category.lower().split('_').pop(0)}.pdf"),
        dpi=300,
        facecolor="w",
        bbox_inches="tight",
    )
    plt.show()
