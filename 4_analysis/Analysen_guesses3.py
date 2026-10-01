# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 01:10:47 2025

@author: serge
"""
# Consolidated imports
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import t, ttest_ind, ttest_rel
from statsmodels.iolib.summary2 import summary_col


def ols_models_to_latex(
    models,
    model_names=None,
    depvar="Dependent variable",
    title="OLS regression results",
    label="tab:ols_results",
    se_note="Heteroskedasticity-robust standard errors (HC3) in parentheses.",
    stars_note="* p<.10, ** p<.05, *** p<.01",
    use_adjusted_r2=True,
    float_format="%.3f",
    add_info_rows=True,
):
    """
    Export one or multiple statsmodels OLS results objects to a single APA-style LaTeX table.

    Parameters
    ----------
    models : list
        List of fitted statsmodels OLS results (e.g., smf.ols(...).fit(cov_type="HC3")).
    model_names : list[str], optional
        Column names for the models (e.g., ["(1)", "(2)"]). Defaults to "(1)", "(2)", ...
    depvar : str
        Dependent variable label shown under the caption.
    title : str
        Table caption.
    label : str
        LaTeX label for referencing the table.
    se_note : str
        Note describing the standard errors (e.g., HC3, clustered, etc.).
    stars_note : str
        Significance stars note.
    use_adjusted_r2 : bool
        If True, reports Adjusted R^2; otherwise reports R^2.
    float_format : str
        Float format used by summary_col.
    add_info_rows : bool
        If True, adds N and (Adjusted) R^2 rows at the bottom.
    """
    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(models))]

    # Optional footer rows (N, R^2 / Adj. R^2)
    info_dict = None
    if add_info_rows:
        r2_label = "Adjusted R$^2$" if use_adjusted_r2 else "R$^2$"
        info_dict = {
            "N": lambda x: f"{int(x.nobs)}",
            r2_label: (lambda x: f"{x.rsquared_adj:.3f}" if use_adjusted_r2 else f"{x.rsquared:.3f}"),
        }

    summ = summary_col(
        models,
        stars=True,
        float_format=float_format,
        model_names=model_names,
        info_dict=info_dict
    )

    df_tab = summ.tables[0]

    # Convert to LaTeX tabular
    tabular = df_tab.to_latex(escape=False)

    notes = f"{se_note} {stars_note}"

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{title}}}\n"
        "\\vspace{2mm}\n"
        f"\\textit{{Dependent variable:}} {depvar}\\\\\n"
        f"\\label{{{label}}}\n"
        f"{tabular}\n"
        "\\begin{flushleft}\n"
        "\\footnotesize \\textit{Note}. "
        f"{notes}\n"
        "\\end{flushleft}\n"
        "\\end{table}"
    )

    return latex

from statsmodels.iolib.summary2 import summary_col

def cox_snell_pseudo_r2(glm_res):
    """Cox–Snell pseudo R^2 for GLM (requires llnull)."""
    llf = glm_res.llf
    llnull = glm_res.llnull
    n = glm_res.nobs
    return 1.0 - np.exp((2.0 / n) * (llnull - llf))

def glm_panel_to_latex(models, model_names, float_format="%.3f"):
    info_dict = {
        "N": lambda x: f"{int(x.nobs)}",
        "Pseudo R$^2$ (CS)": lambda x: f"{cox_snell_pseudo_r2(x):.3f}",
    }
    summ = summary_col(
        models, stars=True, float_format=float_format,
        model_names=model_names, info_dict=info_dict
    )
    return summ.tables[0].to_latex(escape=False)

def ols_panel_to_latex(models, model_names, float_format="%.3f", use_adjusted_r2=True):
    r2_label = "Adjusted R$^2$" if use_adjusted_r2 else "R$^2$"
    info_dict = {
        "N": lambda x: f"{int(x.nobs)}",
        r2_label: (lambda x: f"{x.rsquared_adj:.3f}" if use_adjusted_r2 else f"{x.rsquared:.3f}"),
    }
    summ = summary_col(
        models, stars=True, float_format=float_format,
        model_names=model_names, info_dict=info_dict
    )
    return summ.tables[0].to_latex(escape=False)

def perf_models_to_latex_table(
    logit_models,
    ols_models,
    logit_names,
    ols_names,
    title="Performance regressions",
    label="tab:perf_models",
    depvar_line="Panel A: Success (Logit); Panel B: Inverse success (OLS)",
    se_note="Heteroskedasticity-robust standard errors (HC3) in parentheses.",
    stars_note="* p<.10, ** p<.05, *** p<.01",
    float_format="%.3f",
    use_adjusted_r2=True,
):
    tab_logit = glm_panel_to_latex(logit_models, logit_names, float_format=float_format)
    tab_ols = ols_panel_to_latex(ols_models, ols_names, float_format=float_format, use_adjusted_r2=use_adjusted_r2)

    notes = f"{se_note} {stars_note}"

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{title}}}\n"
        "\\vspace{2mm}\n"
        f"\\textit{{Dependent variable:}} {depvar_line}\\\\\n"
        f"\\label{{{label}}}\n"
        "\\vspace{2mm}\n"
        "\\textbf{Panel A: Logistic regression (Success)}\\\\\n"
        f"{tab_logit}\n"
        "\\vspace{4mm}\n"
        "\\textbf{Panel B: OLS regression (Inverse success)}\\\\\n"
        f"{tab_ols}\n"
        "\\begin{flushleft}\n"
        "\\footnotesize \\textit{Note}. "
        f"{notes}\n"
        "\\end{flushleft}\n"
        "\\end{table}"
    )
    return latex


# ===========================
# 1. INPUT
# ===========================
input_path = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\guesses_combined_cleaned_with_originality.csv"
df_guesses2 = pd.read_csv(input_path)


def safe_literal_eval(x):
    """Parse list-like columns that may be stored as strings; return an empty list for missing/invalid cells."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)  # Enforce a consistent list type for downstream logic.
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return []


df_guesses2["clues_list"] = df_guesses2["clues_list"].apply(safe_literal_eval)
df_guesses2["originality_index_list"] = df_guesses2["originality_index_list"].apply(safe_literal_eval)

df_guesses = df_guesses2.copy()

# Remove rows where duplicates were present (rules require duplicates to be fully excluded).
df_guesses = df_guesses[df_guesses["unique"] != 0].copy()


# ===========================
# 2. Aggregation per word + Human/AI
# ===========================
def compute_row_stats(row):
    """Compute per-row originality summary stats; robust to empty lists and NaNs inside the list."""
    orig_list = row["originality_index_list"]
    clue_list = row["clues_list"]

    if not isinstance(orig_list, list) or len(orig_list) == 0:
        return pd.Series({
            "avg_originality": np.nan,
            "sd_originality": np.nan,
            "min_originality": np.nan,
            "max_originality": np.nan,
            "clue_with_max_originality": np.nan
        })

    # Guard against NaNs inside the list to avoid nan-propagation and index lookup issues.
    clean_orig = [v for v in orig_list if pd.notna(v)]
    if len(clean_orig) == 0:
        return pd.Series({
            "avg_originality": np.nan,
            "sd_originality": np.nan,
            "min_originality": np.nan,
            "max_originality": np.nan,
            "clue_with_max_originality": np.nan
        })

    avg_orig = float(np.mean(clean_orig))
    sd_orig = float(np.std(clean_orig, ddof=1)) if len(clean_orig) > 1 else 0.0
    min_orig = float(np.min(clean_orig))
    max_orig = float(np.max(clean_orig))

    # Align the max originality back to the corresponding clue via the original list index.
    try:
        idx_max = orig_list.index(max_orig)
        clue_max = clue_list[idx_max] if isinstance(clue_list, list) and len(clue_list) > idx_max else np.nan
    except ValueError:
        clue_max = np.nan

    return pd.Series({
        "avg_originality": avg_orig,
        "sd_originality": sd_orig,
        "min_originality": min_orig,
        "max_originality": max_orig,
        "clue_with_max_originality": clue_max
    })


stats_df = df_guesses.apply(compute_row_stats, axis=1)
df_guesses = pd.concat([df_guesses, stats_df], axis=1)

df_guesses["human_label"] = df_guesses["human"].map({0: "AI", 1: "Human"})

# Count how many mystery words per group have at least one clue.
n_with_clues = (
    df_guesses[df_guesses["num_clues"] > 0]
    .groupby("human_label")["mystery_word_id"]
    .count()
)

print("Number of mystery words with >= 1 clue (num_clues > 0):")
print(n_with_clues)



#############################################################
# Descriptive table: number of clues distribution (df_guesses)
#############################################################

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=df_guesses,
    x="num_clues",
    y="mystery_word_id",
    hue="human_label",
    estimator=lambda x: len(x),
    errorbar=None
)

plt.xlabel("Number of Clues")
plt.ylabel("Count")
plt.title("Distribution of Number of Clues (Human vs AI)")
plt.legend(title="")

for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.tight_layout()
plt.show()


#############################################################
# Originality descriptives and paired comparisons (played only)
#############################################################

# Keep a single "played" view: attempts_used is NaN when the word was not playable (e.g., no clues).
df_played = df_guesses[df_guesses["attempts_used"].notna()].copy()

# Binary success: 1 = guessed within 3 attempts, 0 = not guessed.
df_played["success"] = (df_played["attempts_used"] > 0).astype(int)


# ---- Descriptive table (Human vs AI)
orig_table = (
    df_played
    .groupby("human_label")
    .agg(
        mean_avg_originality=("avg_originality", "mean"),
        sd_avg_originality=("avg_originality", "std"),
        mean_sd_originality=("sd_originality", "mean"),
        sd_sd_originality=("sd_originality", "std"),
        n=("mystery_word_id", "count"),
    )
    .round(3)
)

print("\nOriginality descriptives (played words only):")
print(orig_table)


# ---- KDE: avg_originality distribution
plt.figure(figsize=(10, 5))

ax = sns.kdeplot(
    data=df_played,
    x="avg_originality",
    hue="human_label",
    common_norm=False,
    fill=True,
    alpha=0.4,
    legend=True
)

plt.title("Distribution of Average Originality of Clue bundles (Word level)")
plt.xlabel("Average Originality")
plt.ylabel("Density")

# Force legend rendering and remove title
leg = ax.get_legend()
if leg is not None:
    leg.set_title(None)

plt.tight_layout()
plt.show()


# ---- KDE: sd_originality distribution
plt.figure(figsize=(10, 5))

ax = sns.kdeplot(
    data=df_played,
    x="sd_originality",
    hue="human_label",
    common_norm=False,
    fill=True,
    alpha=0.4,
    legend=True
)

plt.title("Distribution of Originality Standard Deviation of Clue bundles (Word level)")
plt.xlabel("Standard Deviation of Originality")
plt.ylabel("Density")

leg = ax.get_legend()
if leg is not None:
    leg.set_title(None)

plt.tight_layout()
plt.show()



# ---- Helper: paired t-test + 95% CI for mean difference (AI - Human)
def paired_ttest_with_ci(df, value_col, id_col="mystery_word_id",
                         group_col="human_label", g1="AI", g2="Human", alpha=0.05):
    """
    Paired t-test comparing g1 vs g2 within the same id_col.
    Returns t, p, mean_diff (g1-g2), 95% CI, and number of paired observations.
    """
    wide = (
        df.pivot(index=id_col, columns=group_col, values=value_col)
        .dropna(subset=[g1, g2])
    )

    x1 = wide[g1].to_numpy()
    x2 = wide[g2].to_numpy()
    diff = x1 - x2
    n = diff.size

    t_stat, p_val = ttest_rel(x1, x2)

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)

    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {"t": t_stat, "p": p_val, "mean_diff": mean_diff,
            "ci_low": ci_low, "ci_high": ci_high, "n_pairs": n}


# ---- Paired t-test: avg_originality
res_avg = paired_ttest_with_ci(df_played, "avg_originality")
print("\nPaired t-test (AI - Human, avg_originality; played only):")
print(f"t = {res_avg['t']:.4f}")
print(f"p = {res_avg['p']:.6g}")
print(f"mean difference (AI - Human) = {res_avg['mean_diff']:.3f}")
print(f"95% CI = [{res_avg['ci_low']:.3f}, {res_avg['ci_high']:.3f}]")
print(f"n pairs = {res_avg['n_pairs']}")


# ---- Paired t-test: sd_originality
res_sd = paired_ttest_with_ci(df_played, "sd_originality")
print("\nPaired t-test (AI - Human, sd_originality; played only):")
print(f"t = {res_sd['t']:.4f}")
print(f"p = {res_sd['p']:.6g}")
print(f"mean difference (AI - Human) = {res_sd['mean_diff']:.4f}")
print(f"95% CI = [{res_sd['ci_low']:.4f}, {res_sd['ci_high']:.4f}]")
print(f"n pairs = {res_sd['n_pairs']}")



#############################################################
# Success metrics (binary + inverse attempts)
#############################################################
#############################################################
# Descriptive table: attempts_used distribution (df_played)
#############################################################

attempt_order = [1, 2, 3, 0]
attempt_labels = {
    1: "1st attempt",
    2: "2nd attempt",
    3: "3rd attempt",
    0: "Not guessed"
}

attempt_table = (
    df_played
    .groupby(["human_label", "attempts_used"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=attempt_order)
)

# Convert to percentages within group
attempt_table_pct = attempt_table.div(attempt_table.sum(axis=1), axis=0) * 100

# Rename columns for readability
attempt_table_pct = attempt_table_pct.rename(columns=attempt_labels).round(1)
attempt_table_cnt = attempt_table.rename(columns=attempt_labels)

print("\nAttempts used – percentage distribution:")
print(attempt_table_pct)

print("\nAttempts used – absolute counts:")
print(attempt_table_cnt)


#############################################################
# Descriptive plot: attempts_used distribution 
#############################################################
# Column and row order
col_order = ["1st attempt", "2nd attempt", "3rd attempt", "Not guessed"]
row_order = ["Human", "AI"]

# Use percentage table only
pct = attempt_table_pct[col_order].reindex(row_order)

x = np.arange(len(col_order))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot percentages
bars_h = ax.bar(
    x - width / 2,
    pct.loc["Human"].values,
    width,
    label="Human",
    color="tab:orange"
)

bars_a = ax.bar(
    x + width / 2,
    pct.loc["AI"].values,
    width,
    label="AI",
    color="tab:blue"
)

ax.set_xlabel("Attempts used")
ax.set_ylabel("Percentage")
ax.set_title("Distribution of Attempts Used (Human vs AI)")
ax.set_xticks(x)
ax.set_xticklabels(col_order)
ax.legend(title="")  # no legend title
ax.set_ylim(0, 100)

# Optional: percentage labels on bars (recommended)
for bars in [bars_h, bars_a]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.tight_layout()
plt.show()


#############################################################
# Success metrics (binary + inverse attempts)
#############################################################

# ------------------------------------------------------------
# A. Overall success probability
# ------------------------------------------------------------
overall_success = df_played["success"].mean()
print("Overall success probability:", round(overall_success, 3))

success_by_human = df_played.groupby("human_label")["success"].mean().round(3)
print("\nSuccess probability by Human vs AI:")
print(success_by_human)

attempt_dist = (
    df_played.groupby("human_label")["attempts_used"]
    .value_counts(normalize=True)
    .sort_index()
)
print("\nDistribution of attempts_used (relative):")
print(attempt_dist.round(3))


# ------------------------------------------------------------
# B. Success probability conditional on number of clues
# ------------------------------------------------------------
success_by_clues = (
    df_played.groupby("num_clues")["success"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "P_success"})
)
print("\nSuccess probability conditional on number of clues:")
print(success_by_clues)

success_by_clues_human = (
    df_played.groupby(["num_clues", "human_label"])["success"]
    .mean()
    .unstack()
    .round(3)
)
print("\nP(success | clues, human vs AI):")
print(success_by_clues_human)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_played,
    x="num_clues",
    y="success",
    hue="human_label",
    estimator="mean",
)
plt.title("Success probability by number of clues")
plt.ylabel("P(success)")
plt.xlabel("Number of clues")
plt.show()


# ------------------------------------------------------------
# Inverse success metric: 1/attempts_used (0 if not guessed)
# ------------------------------------------------------------
def compute_inverse_success(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return 1.0 / x

df_played["p_success_inverse"] = df_played["attempts_used"].apply(compute_inverse_success)

overall_p = df_played["p_success_inverse"].mean()
print("\nOverall inverse success probability:", round(overall_p, 3))

inverse_success_by_human = (
    df_played.groupby("human_label")["p_success_inverse"]
    .mean()
    .round(3)
)
print("\nInverse success probability by Human vs AI:")
print(inverse_success_by_human)

inverse_success_by_clues = (
    df_played.groupby("num_clues")["p_success_inverse"]
    .mean()
    .round(3)
)
print("\nP_inverse(success | number of clues):")
print(inverse_success_by_clues)

inverse_success_by_clues_human = (
    df_played.groupby(["num_clues", "human_label"])["p_success_inverse"]
    .mean()
    .unstack()
    .round(3)
)
print("\nP_inverse(success | clues, human vs AI):")
print(inverse_success_by_clues_human)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_played,
    x="num_clues",
    y="p_success_inverse",
    hue="human_label",
    estimator="mean",
)
plt.title("Inverse success probability by number of clues")
plt.ylabel("P_success_inverse = 1 / attempts_used")
plt.xlabel("Number of clues")
plt.show()

#############################################################
# Summary: attempts_used (successful guesses only)
#############################################################

df_attempts_success = df_played[df_played["attempts_used"] > 0]

attempts_summary = (
    pd.concat([
        df_attempts_success.assign(group="Overall"),
        df_attempts_success.assign(group=df_attempts_success["human_label"])
    ])
    .groupby("group")["attempts_used"]
    .agg(mean_attempts="mean", n_words="count")
    .round(3)
)

print("\nAverage number of attempts (successful guesses only):")
print(attempts_summary)



#############################################################
# Summary: number of clues (words with at least one clue)
#############################################################

df_with_clues = df_guesses[df_guesses["num_clues"] > 0]

clues_summary = (
    pd.concat([
        df_with_clues.assign(group="Overall"),
        df_with_clues.assign(group=df_with_clues["human_label"])
    ])
    .groupby("group")["num_clues"]
    .agg(mean_clues="mean", n_words="count")
    .round(3)
)

print("\nClue count summary (words with ≥ 1 clue):")
print(clues_summary)




# ======================================================
# Empirical attempt distribution and heatmap
# ======================================================
def empirical_attempt_distribution(group_df):
    """Return P(attempts_used = k | num_clues = n) as a table (rows: num_clues, columns: attempts_used)."""
    dist = (
        group_df.groupby(["num_clues", "attempts_used"])
        .size()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
        .rename(columns={0: "not_guessed", 1: "first_try", 2: "second_try", 3: "third_try"})
        .sort_index()
    )
    return dist

human_dist = empirical_attempt_distribution(df_played[df_played["human_label"] == "Human"])
ai_dist    = empirical_attempt_distribution(df_played[df_played["human_label"] == "AI"])

print("\nHuman distribution:")
print(human_dist.round(3))
print("\nAI distribution:")
print(ai_dist.round(3))


def empirical_attempt_heatmap_matrix(group_df):
    """
    Matrix format for heatmaps:
    rows    = outcome (not_guessed, first_try, second_try, third_try)
    columns = num_clues
    cells   = P(outcome | num_clues, group)
    """
    dist = (
        group_df.groupby(["num_clues", "attempts_used"])
        .size()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
    )

    # Ensure all outcome columns exist to keep a stable layout across groups.
    for col in [0, 1, 2, 3]:
        if col not in dist.columns:
            dist[col] = 0.0
    dist = dist[[0, 1, 2, 3]]

    dist = dist.T
    dist.index = ["not_guessed", "first_try", "second_try", "third_try"]
    dist = dist.sort_index(axis=1)
    return dist


human_heat = empirical_attempt_heatmap_matrix(df_played[df_played["human_label"] == "Human"])
ai_heat    = empirical_attempt_heatmap_matrix(df_played[df_played["human_label"] == "AI"])

# Display order: bottom-to-top (Not guessed -> 3rd -> 2nd -> 1st) for interpretability.
human_heat = human_heat.iloc[::-1]
ai_heat    = ai_heat.iloc[::-1]

print("\nHuman heatmap matrix:")
print(human_heat.round(3))
print("\nAI heatmap matrix:")
print(ai_heat.round(3))

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(human_heat, annot=True, fmt=".2f", cmap="Blues")
plt.title("P(attempt | number of clues) – Human")
plt.xlabel("Number of clues")
plt.ylabel("Attempt")
plt.yticks([0.5, 1.5, 2.5, 3.5], ["Not guessed", "3rd try", "2nd try", "1st try"])

plt.subplot(1, 2, 2)
sns.heatmap(ai_heat, annot=True, fmt=".2f", cmap="Blues")
plt.title("P(attempt | number of clues) – AI")
plt.xlabel("Number of clues")
plt.ylabel("Attempt")
plt.yticks([0.5, 1.5, 2.5, 3.5], ["Not guessed", "3rd try", "2nd try", "1st try"])

plt.tight_layout()
plt.show()



#############################################################
# Originality vs number of clues 

df_orig = df_guesses[df_guesses["num_clues"] > 0].copy()

#############################################################
# Linear regression with robust standard errors (HC3)
# avg_originality ~ num_clues * human_label
#############################################################

import statsmodels.formula.api as smf

model_0 = smf.ols(
    "avg_originality ~ num_clues",
    data=df_orig
).fit(cov_type="HC3")

print(model_0.summary())

model_1= smf.ols(
    "avg_originality ~ num_clues + human_label",
    data=df_orig
).fit(cov_type="HC3")

print(model_1.summary())

model_2= smf.ols(
    "avg_originality ~ num_clues * human_label",
    data=df_orig
).fit(cov_type="HC3")

print(model_2.summary())


latex_code = ols_models_to_latex(
    models=[model_0, model_1, model_2],
    model_names=["(1)", "(2)" , "(3)"],
    depvar="Average originality",
    title="Determinants of average originality of clue bundles",
    label="tab:orig_models",
    se_note="Heteroskedasticity-robust standard errors (HC3) in parentheses.",
    use_adjusted_r2=True
)

print(latex_code)


#############################################################
# Logistic regression (GLM Binomial) with robust SE (HC3)
# success ~ human_label + num_clues + avg_originality (+ interactions)
#############################################################

import statsmodels.api as sm
import statsmodels.formula.api as smf


# Use only playable words (attempts_used is NaN when no clues existed / not playable)
df_played = df_guesses[df_guesses["attempts_used"].notna()].copy()

# Binary outcome: 1 = guessed within 3 attempts, 0 = not guessed
df_played["success"] = (df_played["attempts_used"] > 0).astype(int)

#effects model1
logit_1 = smf.glm(
    formula="success ~ human_label",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

print(logit_1.summary())

#effects model2
logit_2 = smf.glm(
    formula="success ~ human_label + avg_originality",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

print(logit_2.summary())

#effects model3
logit_3 = smf.glm(
    formula="success ~ human_label + num_clues",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

print(logit_3.summary())


# Main-effects model
logit_main = smf.glm(
    formula="success ~ human_label + avg_originality+ num_clues",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

print(logit_main.summary())

# Interaction model (recommended next step)
logit_int = smf.glm(
    formula="success ~ human_label * avg_originality + human_label * num_clues",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

print(logit_int.summary())

#############################################################
# Secondary outcome model: p_success_inverse ~ same regressors (OLS + HC3)
#############################################################

df_played["p_success_inverse"] = df_played["attempts_used"].apply(
    lambda x: 0.0 if x == 0 else 1.0 / x
)

ols_inv_1 = smf.ols(
    "p_success_inverse ~ human_label",
    data=df_played
).fit(cov_type="HC3")

print(ols_inv_1.summary())

ols_inv_2 = smf.ols(
    "p_success_inverse ~ human_label + avg_originality",
    data=df_played
).fit(cov_type="HC3")

print(ols_inv_2.summary())

ols_inv_3 = smf.ols(
    "p_success_inverse ~ human_label + num_clues",
    data=df_played
).fit(cov_type="HC3")

print(ols_inv_3.summary())

ols_inv_main = smf.ols(
    "p_success_inverse ~ human_label + avg_originality + num_clues",
    data=df_played
).fit(cov_type="HC3")

print(ols_inv_main.summary())

ols_inv_int = smf.ols(
    "p_success_inverse ~ human_label * avg_originality + human_label * num_clues",
    data=df_played
).fit(cov_type="HC3")

print(ols_inv_int.summary())




latex = perf_models_to_latex_table(
    logit_models=[logit_1, logit_2, logit_3, logit_main, logit_int],
    ols_models=[ols_inv_1, ols_inv_2, ols_inv_3, ols_inv_main, ols_inv_int],
    logit_names=["(1) Human", "(2) +Avg. Originality","(3) +Number Clues", "(4) 2+3 ","(5) + Interactions"],
    ols_names=["(1) Human", "(2) +Avg. Originality","(3) +Number Clues", "(4) 2+3 ","(5) + Interactions"],
    title="Usefullness models: success and inverse success",
    label="tab:perf_success_inverse",
    se_note="Heteroskedasticity-robust standard errors (HC3) in parentheses."
)

print(latex)


#############################################################
# CATEGORICAL num_clues (only 1, 2, 4) for Logit + OLS (HC3)
# + quadratic term for average originality
#############################################################


# Use only playable words (attempts_used is NaN when no clues existed / not playable)
df_played = df_guesses[df_guesses["attempts_used"].notna()].copy()

# Binary outcome: 1 = guessed within 3 attempts, 0 = not guessed
df_played["success"] = (df_played["attempts_used"] > 0).astype(int)

# Inverse success outcome
df_played["p_success_inverse"] = df_played["attempts_used"].apply(
    lambda x: 0.0 if x == 0 else 1.0 / x
)

# Ensure num_clues is treated as categorical with a stable ordering (reference = 1 clue)
df_played["num_clues_cat"] = pd.Categorical(
    df_played["num_clues"],
    categories=[1, 2, 4],
    ordered=True
)

df_played["avg_originality_sq"] = df_played["avg_originality"] ** 2
"""
# Center originality to reduce multicollinearity with the squared term
#df_played["avg_orig_c"] = df_played["avg_originality"] - df_played["avg_originality"].mean()
#df_played["avg_orig_c_sq"] = df_played["avg_orig_c"] ** 2
"""


# ------------------------------------------------------------
# LOGIT MODELS (GLM Binomial) with categorical num_clues
# ------------------------------------------------------------

# Model (1): Human only
logit_cat_1 = smf.glm(
    formula="success ~ human_label",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

# Model (2): + Avg. originality (linear)
logit_cat_2 = smf.glm(
    formula="success ~ human_label + avg_originality",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

# Model (3): + Categorical number of clues
logit_cat_3 = smf.glm(
    formula="success ~ human_label + C(num_clues_cat)",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

# Model (4): Full main-effects model (linear originality)
logit_cat_main = smf.glm(
    formula="success ~ human_label + avg_originality + C(num_clues_cat)",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")

# Model (5): Quadratic originality + categorical clues (recommended next step)
# Using centered originality terms for numerical stability
logit_cat_quad = smf.glm(
    formula="success ~ human_label +  avg_originality + avg_originality_sq + C(num_clues_cat)",
    data=df_played,
    family=sm.families.Binomial()
).fit(cov_type="HC3")


# ------------------------------------------------------------
# OLS MODELS with categorical num_clues (HC3)
# ------------------------------------------------------------

# Model (1): Human only
ols_inv_cat_1 = smf.ols(
    "p_success_inverse ~ human_label",
    data=df_played
).fit(cov_type="HC3")

# Model (2): + Avg. originality (linear)
ols_inv_cat_2 = smf.ols(
    "p_success_inverse ~ human_label + avg_originality",
    data=df_played
).fit(cov_type="HC3")

# Model (3): + Categorical number of clues
ols_inv_cat_3 = smf.ols(
    "p_success_inverse ~ human_label + C(num_clues_cat)",
    data=df_played
).fit(cov_type="HC3")

# Model (4): Full main-effects model (linear originality)
ols_inv_cat_main = smf.ols(
    "p_success_inverse ~ human_label + avg_originality + C(num_clues_cat)",
    data=df_played
).fit(cov_type="HC3")

# Model (5): Quadratic originality + categorical clues
ols_inv_cat_quad = smf.ols(
    "p_success_inverse ~ human_label + avg_originality + avg_originality_sq + C(num_clues_cat)",
    data=df_played
).fit(cov_type="HC3")


# ------------------------------------------------------------
# Print summaries (optional)
# ------------------------------------------------------------
print(logit_cat_quad.summary())
print(ols_inv_cat_quad.summary())


# ------------------------------------------------------------
# LaTeX export
# ------------------------------------------------------------
latex = perf_models_to_latex_table(
    logit_models=[logit_cat_1, logit_cat_2, logit_cat_3, logit_cat_main, logit_cat_quad],
    ols_models=[ols_inv_cat_1, ols_inv_cat_2, ols_inv_cat_3, ols_inv_cat_main, ols_inv_cat_quad],
    logit_names=["(1) Human", "(2) + Avg. Originality", "(3) + Clues (cat)", "(4) 2+3", "(5) + Orig.^2"],
    ols_names=["(1) Human", "(2) + Avg. Originality", "(3) + Clues (cat)", "(4) 2+3", "(5) + Orig.^2"],
    title="Usefulness models: success and inverse success (categorical clue counts)",
    label="tab:perf_success_inverse_cat_quad",
    se_note="Heteroskedasticity-robust standard errors (HC3) in parentheses."
)

print(latex)



# ------------------------------------------------------------
# Data for Logic reg test
# ------------------------------------------------------------


mean_orig = df_played["avg_originality"].mean()

pred_df = pd.DataFrame({
    "human_label": ["AI", "AI", "AI"],
    "num_clues_cat": [1, 2, 4],
    "avg_originality": mean_orig
})

pred_df["pred_prob"] = logit_cat_main.predict(pred_df)
print(pred_df)
