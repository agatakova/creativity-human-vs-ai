"""
Created on Mon Nov 17 18:38:18 2025

@author: serge
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind

from scipy.stats import t, ttest_rel
import numpy as np


import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f

from collections import Counter

from statsmodels.iolib.summary2 import summary_col

# Feste Farben pro Modell 
ai_colors = {
    "ChatGPT":  "#1f77b4",  # blau
    "DeepSeek": "#ff7f0e",  # orange
    "Gemini":   "#2ca02c",  # grün
    "Claude":   "#d62728",  # rot
}


# ==============================
# 1) Daten einlesen
# ==============================
df = pd.read_csv(
    r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\data_merged_clues_ai_with_originality.csv"
)

print("Erste Zeilen:")
print(df.head())
print("\nSpaltennamen:", df.columns.tolist())

# ==============================
# 2) Grundlegende Checks & rohe Counts
# ==============================
print("\n--- Rohe Verteilung human (0/1) ---")
print(df["human"].value_counts())

print("\n--- Missing Values originality_index (roh) ---")
print(df["originality_index"].isna().sum())

# ==============================
# 3) Cleaned DF: Zeilen ohne originality_index entfernen
# ==============================
df_clean = df.dropna(subset=["originality_index"]).copy()

print("\nAnzahl Zeilen vor dem Droppen:", len(df))
print("Anzahl Zeilen nach dem Droppen:", len(df_clean))

# optional: clean Data auch speichern
df_clean.to_csv("datensatz_clean.csv", index=False)

print("\n--- Verteilung human (0/1) ---")
print(df_clean["human"].value_counts())


# ==============================
# 4) Human-Label hinzufügen (AI / Human)
# ==============================
df_clean["human_label"] = df_clean["human"].map({0: "AI", 1: "Human"})

print("\n--- Verteilung human (cleaned) ---")
print(df_clean["human"].value_counts().rename({0: "AI", 1: "Human"}))

df_clean["ai_temperature"] = pd.to_numeric(df_clean["ai_temperature"], errors="coerce")


# ==============================
# 5) Deskriptive Statistiken nach Human vs AI
# ==============================
summary = (
    df_clean
    .groupby("human_label")["originality_index"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .rename(columns={
        "count": "N",
        "mean": "Mean",
        "median": "Median",
        "std": "StdAbw",
        "min": "Min",
        "max": "Max"
    })
)

print("\n=== Deskriptive Statistiken: Originality nach Human vs AI ===")
print(summary)

# ==============================
# 6) t-Test: Unterschied Human vs AI in originality_index
# ==============================
human_vals = df_clean.loc[df_clean["human"] == 1, "originality_index"].dropna()
ai_vals    = df_clean.loc[df_clean["human"] == 0, "originality_index"].dropna()


print("\n--- Gruppen-Größen für t-Test ---")
print("N Human:", len(human_vals))
print("N AI   :", len(ai_vals))

if len(human_vals) < 2 or len(ai_vals) < 2:
    print("\n⚠️ Zu wenige Beobachtungen in einer der Gruppen für einen sinnvollen t-Test.")
else:
    # t-Test
    t_stat, p_val = ttest_ind(human_vals, ai_vals, equal_var=False)

    # Mittelwerte
    mean_h = np.mean(human_vals)
    mean_ai = np.mean(ai_vals)
    diff = mean_h - mean_ai

    # Standardfehler für Welch-Test
    se = np.sqrt(np.var(human_vals, ddof=1)/len(human_vals) +
                 np.var(ai_vals, ddof=1)/len(ai_vals))

    # Welch-Satterthwaite Freiheitsgrade
    se_h = np.var(human_vals, ddof=1)/len(human_vals)
    se_ai = np.var(ai_vals, ddof=1)/len(ai_vals)

    df = (se_h + se_ai)**2 / ((se_h**2)/(len(human_vals)-1) +
                              (se_ai**2)/(len(ai_vals)-1))

    # 95%-Konfidenzintervall
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    print("\n=== t-Test: Originality Human vs AI ===")
    print("t-Statistik:", t_stat)
    print("p-Wert     :", p_val)
    print("Mittelwert Human:", mean_h)
    print("Mittelwert AI   :", mean_ai)
    print("Differenz (Human - AI):", diff)
    print(f"95% Konfidenzintervall der Differenz: [{ci_lower:.4f}, {ci_upper:.4f}]")
    


def ols_to_latex_apa(models,
                     model_names=None,
                     depvar="Originality Index",
                     title="OLS regression results",
                     label="tab:ols_results",
                     notes="Robust standard errors clustered at the mystery word level in parentheses. * p<.10, ** p<.05, *** p<.01"):

    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(models))]

    # 1) summary_col erzeugen
    summ = summary_col(
        models,
        stars=True,
        float_format="%.3f",
        model_names=model_names,
        info_dict={
            "N":   lambda x: f"{int(x.nobs)}",
            "R²":  lambda x: f"{x.rsquared:.3f}",
        }
    )

    # 2) DataFrame extrahieren
    df_tab = summ.tables[0]

    # 3) Zeilen umbenennen (APA-Style)
    df_tab.rename(index={
        "R-squared": "R²",
        "R-squared Adj.": "Adjusted R²",
        "R²": "R²",          # optional: falls du es beibehalten willst
        "N": "N"
    }, inplace=True)

    # 4) Tabular erstellen
    tabular = df_tab.to_latex(escape=False)

    # 5) APA-kompatible Table-Umgebung
    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{" + title + "}\n"
        "\\vspace{2mm}\n"
        "\\textit{Dependent variable:} " + depvar + "\\\\\n"
        "\\label{" + label + "}\n"
        + tabular + "\n"
        "\\begin{flushleft}\n"
        "\\footnotesize \\textit{Note}. " + notes + "\n"
        "\\end{flushleft}\n"
        "\\end{table}"
    )

    return latex


#############################################################################
# AI_models Setting overview
#############################################################################

df_clean["ai_label"] = df_clean["ai_id"].map({
    1: "ChatGPT",
    2: "DeepSeek",
    3: "Gemini",
    4: "Claude"
})

df_clean["model_label"] = df_clean["ai_label"].fillna("Human")


import numpy as np
import pandas as pd

df_overview = df_clean.copy()

# Ensure numeric temperature
df_overview["ai_temperature"] = pd.to_numeric(df_overview["ai_temperature"], errors="coerce")

# Flash: NA if missing, Flash if == 1
df_overview["flash_label"] = np.where(
    df_overview["gemini_flash"].isna(),
    "NA",
    np.where(df_overview["gemini_flash"] == 1, "Flash", "Pro")
)

# Full memory: NA if missing
df_overview["memory_label"] = np.where(
    df_overview["full_memory"].isna(),
    "NA",
    np.where(df_overview["full_memory"] == 1, "FullMemory", "NoMemory")
)

# Anti-anchoring: NA if missing
df_overview["aa_label"] = np.where(
    df_overview["anti_anchoring_prompt"].isna(),
    "NA",
    np.where(df_overview["anti_anchoring_prompt"] == 1, "AntiAnchor", "NoAntiAnchor")
)

# Compact combined setting label (readable but explicit)
df_overview["setting_label"] = (
    df_overview["memory_label"]
    + " | "
    + df_overview["aa_label"]
    + " | "
    + df_overview["flash_label"]
)

word_counts = (
    df_overview
    .groupby(
        ["ai_temperature", "model_label", "setting_label"],
        dropna=False
    )
    .agg(
        n_words=("mystery_word_id", "nunique")
    )
    .reset_index()
    .sort_values(["ai_temperature", "model_label", "setting_label"])
)

print(word_counts)


overview_counts = (
    word_counts
    .pivot_table(
        index="ai_temperature",
        columns=["model_label", "setting_label"],
        values="n_words",
        fill_value=0
    )
    .sort_index()
)

print("\nOverview: number of mystery words per temperature × model × setting")
print(overview_counts)



#########################Latex 
def latex_escape(s: str) -> str:
    """Escape LaTeX-sensitive characters (minimal set, keeps it readable)."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s

# overview_counts: index=ai_temperature, columns=MultiIndex(model_label, setting_label), values=n_words
# --- formatting for LaTeX table output ---

tbl = overview_counts.copy()

# Round temperature index to one decimal
tbl.index = tbl.index.map(lambda x: round(float(x), 1) if pd.notna(x) else x)

# Ensure counts are integers
tbl = tbl.round(0).astype("Int64")  # keeps NA as <NA>, not 0

# Optional: sort temperatures numerically after rounding
tbl = tbl.sort_index()


# Build two-line headers using \makecell{Model\\Setting}
new_cols = []
for model, setting in tbl.columns:
    model_esc = latex_escape(model)
    setting_esc = latex_escape(setting)
    new_cols.append(r"\makecell{" + model_esc + r"\\" + setting_esc + "}")
tbl.columns = new_cols

# Index name (optional)
tbl.index.name = "Temperature"

# Export as LaTeX (no escaping, because we already escaped and inserted LaTeX)
latex_body = tbl.to_latex(
    escape=False,
    na_rep="",
    column_format="l" + "r" * tbl.shape[1],
    bold_rows=False
)

print(latex_body)



#############################################################################
# OLS originality  human
#############################################################################
df_clean["ai_label"] = df_clean["ai_id"].map({
    1: "ChatGPT",
    2: "DeepSeek",
    3: "Gemini",
    4: "Claude"
})

df_clean["model_label"] = df_clean["ai_label"].fillna("Human")


'''
test_human = smf.ols("originality_index ~ human", data=df_clean).fit()
print(test_human.summary())

robust_test_human = smf.ols("originality_index ~ human_label", data=df_clean).fit(
    cov_type="HC3"
)
print(robust_test_human.summary())

cluster_test_human = smf.ols("originality_index ~ human_label", data=df_clean).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_clean["mystery_word_id"]}
)
print(cluster_test_human.summary())
'''

robust_test_human = smf.ols(
    "originality_index ~ C(human_label, Treatment(reference='Human'))",
    data=df_clean
).fit(
    cov_type="HC3"
)
print(robust_test_human.summary())

cluster_test_human = smf.ols(
    "originality_index ~ C(human_label, Treatment(reference='Human'))",
    data=df_clean
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_clean["mystery_word_id"]}
)
print(cluster_test_human.summary())


robust_test_human_ai_models = smf.ols(
    "originality_index ~ C(model_label, Treatment(reference='Human'))",
    data=df_clean
).fit(
    cov_type="HC3",
)
print(robust_test_human_ai_models.summary())

cluster_test_human_ai_models = smf.ols(
    "originality_index ~ C(model_label, Treatment(reference='Human'))",
    data=df_clean
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_clean["mystery_word_id"]}
)
print(cluster_test_human_ai_models.summary())


latex_code = ols_to_latex_apa(
    models=[cluster_test_human, cluster_test_human_ai_models],
    model_names=["(1) Human vs AI", "(2) Humans vs AI-Models"],
    depvar="Originality Index",
    title="Effect of Human vs AI on Originality",
    label="tab:orig_ols"
)

print(latex_code)


#############################################################################
#only AI, only full memory ==1, no const
#############################################################################
df_ai_only = df_clean[df_clean["ai_id"].notna()].copy()
df_ai_full_memory_only = df_ai_only[df_ai_only["full_memory"] == 1].copy()

robust_test_ai_models_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label)",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3",
)
print(robust_test_ai_models_fullmemory_nc.summary())

cluster_test_ai_models_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label)",
    data=df_ai_full_memory_only
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_ai_full_memory_only["mystery_word_id"]}
)
print(cluster_test_ai_models_fullmemory_nc.summary())

#############################################################################
robust_test_ai_models_temp_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label) + ai_temperature",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3",
)
print(robust_test_ai_models_temp_fullmemory_nc.summary())

cluster_test_ai_models_temp_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label) + ai_temperature",
    data=df_ai_full_memory_only
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_ai_full_memory_only["mystery_word_id"]}
)
print(cluster_test_ai_models_temp_fullmemory_nc.summary())

#############################################################################
robust_test_ai_models_temp_int_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label) + C(model_label):ai_temperature",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3",
)
print(robust_test_ai_models_temp_int_fullmemory_nc.summary())

cluster_test_ai_models_temp_int_fullmemory_nc = smf.ols(
    "originality_index ~ -1 + C(model_label) + C(model_label):ai_temperature",
    data=df_ai_full_memory_only
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_ai_full_memory_only["mystery_word_id"]}
)
print(cluster_test_ai_models_temp_int_fullmemory_nc.summary())      

latex_code = ols_to_latex_apa(
    models=[cluster_test_ai_models_fullmemory_nc,cluster_test_ai_models_temp_fullmemory_nc, cluster_test_ai_models_temp_int_fullmemory_nc],
    model_names=["(1) Models", "(2) +Temperature", "(3)+Interaction"],
    depvar="Originality Index",
    title="Effect AI Temperature in differentg Models on Originality",
    label="tab:orig_ols"
)

print(latex_code)


#
# ==============================
# 7) Verteilungsplot: KDE Human vs AI
# ==============================

ax = sns.kdeplot(
    data=df_clean,
    x="originality_index",
    hue="human_label",
    fill=True,
    alpha=0.4
)

# Legendentitel entfernen (aber Legende behalten!)
legend = ax.get_legend()
legend.set_title("")

plt.title("Distribution Originality Index: Human vs AI")
plt.xlabel("Originality Index")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

"""
# ==============================
# 8) Violinen+ Boxplot: Originality Human vs AI mit Mean
# ==============================

sns.violinplot(
    data=df_clean,
    x="human_label",
    y="originality_index",
    inner=None,
    alpha=0.6,
    zorder=1
)

sns.boxplot(
    data=df_clean,
    x="human_label",
    y="originality_index",
    width=0.2,
    boxprops={"facecolor": "white", "edgecolor": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    medianprops={"color": "red", "linewidth": 2},
    zorder=2
)

# --- Mean hinzufügen (oben drauf!) ---
group_means = df_clean.groupby("human_label")["originality_index"].mean()

plt.scatter(
    x=[0, 1],                  # Positionen der Gruppen
    y=group_means.values,      # Mean-Werte
    color="blue",
    marker="D",
    s=90,
    zorder=10,                 # Mean nach vorne holen
    label="Mean"
)

plt.legend()
plt.title("Violin + Boxplot with Median (read) und Mean (blue)")
plt.ylabel("Originality Index")
plt.xlabel("")
plt.tight_layout()
plt.show()
"""

# ==============================
# AI-Vergleich
# ==============================


stats_models = (
    df_clean.groupby("ai_label")["originality_index"]
    .agg(["count", "mean", "std", "min", "max"])
    .rename(columns={
        "count": "N",
        "mean": "Mean",
        "std": "StdAbw"
    })
)
print(stats_models)
      
# ==============================
# Verteilungsplot: KDE AI
# ==============================      
ax = sns.kdeplot(
    data=df_clean,
    x="originality_index",
    hue="ai_label",
    fill=True,
    alpha=0.25
)

# Legendentitel entfernen (aber Legende behalten!)
legend = ax.get_legend()
legend.set_title("")

plt.title("Distribution Originality Index: Different AI-Modells")
plt.xlabel("Originality Index")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

"""
# ==============================
# Violinenplot mit boxplot
# ==============================  
#
ax = sns.violinplot(
    data=df_clean,
    x="ai_label",
    y="originality_index",
    inner=None,
    alpha=0.6
)

sns.boxplot(
    data=df_clean,
    x="ai_label",
    y="originality_index",
    width=0.15,
    boxprops={"facecolor": "white", "edgecolor": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    medianprops={"color": "red", "linewidth": 2},
    zorder=5
)

# Mean pro Modell
means = df_clean.groupby("ai_label")["originality_index"].mean()

plt.scatter(
    x=range(len(means)),
    y=means.values,
    color="blue",
    marker="D",
    s=90,
    zorder=10,
    label="Mean"
)

plt.legend()
plt.title("Violin + Boxplot: Originality Different AI-Modells")
plt.ylabel("Originality Index")
plt.xlabel("")
plt.tight_layout()
plt.show()

"""

"""
# ==============================
# Tukney Test
# ============================== 
# df_ai_only without human data

stats.shapiro(df_ai_only["originality_index"])

from scipy.stats import levene

groups = [
    df_ai_only[df_ai_only["ai_label"] == label]["originality_index"]
    for label in df_ai_only["ai_label"].unique()
]

levene(*groups)

#####!!!!!!!!!!!!!!!!!!!!!!!!!
tukey = pairwise_tukeyhsd(
    df_ai_only["originality_index"],
    df_ai_only["ai_label"],
    alpha=0.05
)
print(tukey)


"""
# ==============================
# Temperature tests
# ============================== 

# temp tab
df_ai_only["ai_temperature"].describe()
df_ai_only["ai_temperature"].unique()
df_ai_only["ai_temperature"].value_counts()
df_ai_only["ai_temperature"].value_counts().sort_index()


# temp doesnt explain variations in originality 


model0 = smf.ols("originality_index ~ ai_temperature", data=df_ai_only).fit()
print(model0.summary())
    
robust0 = model0.get_robustcov_results(cov_type="HC3")
print(robust0.summary())

cluster0 = model0.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster0.summary())



# ==============================
# Temperature plot
# ============================== 
# temperature plot
# Counts pro Temperatur × Modell
temp_model_counts = (
    df_ai_only
    .groupby(["ai_temperature", "ai_label"])
    .size()
    .unstack(fill_value=0)
)

# sicherstellen, dass die Spalten in der richtigen Reihenfolge sind
temp_model_counts = temp_model_counts[["ChatGPT", "DeepSeek", "Gemini", "Claude"]]

fig, ax = plt.subplots(figsize=(10, 6))

# X-Achse
x = np.arange(len(temp_model_counts.index))
temps = temp_model_counts.index.astype(str)

bottom = np.zeros(len(temp_model_counts.index))

# Prozentwerte vorbereiten
temp_model_pct = temp_model_counts.div(temp_model_counts.sum(axis=1), axis=0) * 100

# Gestapelte Balken zeichnen
for model in temp_model_counts.columns:
    counts = temp_model_counts[model]
    pct = temp_model_pct[model]

    bars = ax.bar(
        x,
        counts,
        bottom=bottom,
        color=ai_colors[model],
        label=model
    )

    # Prozentwerte in die Balkensegmente schreiben
    for i, (bar, percent) in enumerate(zip(bars, pct)):
        if percent > 2:  # nur drucken wenn > 2% (kannst du anpassen)
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bottom[i] + counts.iloc[i] / 2,
                f"{percent:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold"
            )

    bottom += counts.values  # Stapel hochsetzen

# Achsenbeschriftungen
ax.set_xticks(x)
ax.set_xticklabels(temps)
ax.set_xlabel("Temperature")
ax.set_ylabel("Count")
ax.set_title("Number of Clues from AI-Modelle per Temperature (with %)")
ax.legend(title="AI Model")

plt.tight_layout()
plt.show()



# check corr temp und originality
df_ai_only[["ai_temperature", "originality_index"]].corr()

model_corr = smf.ols("originality_index ~ ai_temperature", data=df_ai_only).fit()
print(model_corr.summary())
robust_model_corr = model_corr.get_robustcov_results(cov_type="HC3")
print(robust_model_corr.summary())

# cluster nach mystery_word_id (empfohlen)
cluster_model_corr = model_corr.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster_model_corr.summary())


# 
latex_code = ols_to_latex_apa(
    models=[cluster_model_corr],
    model_names=["Temperature"],
    depvar="Originality Index",
    title="Originality Index on AI-Temperature Setting",
    label="tab:orig_human_ai",
    notes="Robust standard errors clustered at the mystery word level in parentheses. * p<.10, ** p<.05, *** p<.01"
)

print(latex_code)

# ==============================
# OLS diff model
# ============================== 
# OLS

model1 = smf.ols("originality_index ~ C(ai_label)", data=df_ai_only).fit()
#model = smf.ols("originality_index ~ C(ai_label) + ...", data=df_ai_only).fit()
# robuste SE
# robuste SE
robust1 = model1.get_robustcov_results(cov_type="HC3")
print(robust1.summary())

# cluster nach mystery_word_id (empfohlen)
cluster1 = model1.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster1.summary())



# ==============================
# OLS diff model + temp
# ============================== 
# OLS
# OLS with temp

model2 = smf.ols("originality_index ~ C(ai_label) + ai_temperature", data=df_ai_only).fit()
# robuste SE
# robuste SE
robust2 = model2.get_robustcov_results(cov_type="HC3")
print(robust2.summary())

# cluster nach mystery_word_id (empfohlen)
cluster2 = model2.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster2.summary())

# ==============================
# OLS diff model + temp+ full_memory
# ============================== 
#

model3 = smf.ols("originality_index ~ C(ai_label) + ai_temperature + full_memory", data=df_ai_only).fit()
#model = smf.ols("originality_index ~ C(ai_label) + ...", data=df_ai_only).fit()
# robuste SE
# robuste SE
robust3 = model3.get_robustcov_results(cov_type="HC3")
print(robust3.summary())

# cluster nach mystery_word_id (empfohlen)
cluster3 = model3.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster3.summary())

# ==============================
# OLS diff model +full_memory
# ============================== 
#

model4 = smf.ols("originality_index ~ C(ai_label)+ full_memory", data=df_ai_only).fit()
#model = smf.ols("originality_index ~ C(ai_label) + ...", data=df_ai_only).fit()
# robuste SE
# robuste SE
robust4 = model4.get_robustcov_results(cov_type="HC3")
print(robust4.summary())

# cluster nach mystery_word_id (empfohlen)
cluster4 = model4.get_robustcov_results(
    cov_type="cluster",
    groups=df_ai_only["mystery_word_id"]
)
print(cluster4.summary())



#  LaTeX-Tabelle 
latex_code = ols_to_latex_apa(
    models=[cluster0, cluster1, cluster2, cluster3],
    model_names=["(1) Only Temperature","(2) Different AI-Modells", "(3) + Temperature", "(4) + Full Memory"],
    depvar="Originality Index",
    title="Effect Different AI-Models and Settings on Originality",
    label="tab:orig_human_ai",
    notes="ChatGPT is the omitted reference category; coefficients for the other AIs represent differences relative to ChatGPT. Robust standard errors clustered at the mystery word level in parentheses. * p<.10, ** p<.05, *** p<.01"
)

print(latex_code)


# Jetzt: LaTeX-Tabelle für BEIDE Modelle nebeneinander
latex_code = ols_to_latex_apa(
    models=[ cluster1, cluster2, cluster3],
    model_names=["(1) Different AI-Modells", "(2) + Temperature", "(3) + Full Memory"],
    depvar="Originality Index",
    title="Effect Different AI-Models and Settings on Originality",
    label="tab:orig_human_ai",
    notes="ChatGPT is the omitted reference category; coefficients for the other AIs represent differences relative to ChatGPT. Robust standard errors clustered at the mystery word level in parentheses. * p<.10, ** p<.05, *** p<.01"
)

print(latex_code)


##################################################
# OLS just ChatGPT
##################################################
df_chatgpt = df_ai_only[df_ai_only["ai_label"] == "ChatGPT"].copy()
df_chatgpt.shape

model5 = smf.ols(
    "originality_index ~ ai_temperature",
    data=df_chatgpt
).fit()
robust5 = model5.get_robustcov_results(cov_type="HC3")
print(robust5.summary())

cluster5 = model5.get_robustcov_results(
    cov_type="cluster",
    groups=df_chatgpt["mystery_word_id"]
)
print(cluster5.summary())


model6 = smf.ols(
    "originality_index ~ ai_temperature + full_memory",
    data=df_chatgpt
).fit()
robust6 = model6.get_robustcov_results(cov_type="HC3")
print(robust6.summary())

cluster6 = model6.get_robustcov_results(
    cov_type="cluster",
    groups=df_chatgpt["mystery_word_id"]
)
print(cluster6.summary())


df_chatgpt_2 = df_chatgpt[df_chatgpt["ai_temperature"] == 2.0].copy()
df_chatgpt_2.shape

model6_2 = smf.ols(
    "originality_index ~ full_memory",
    data=df_chatgpt_2
).fit()
robust6_2 = model6_2.get_robustcov_results(cov_type="HC3")
print(robust6_2.summary())



df_chatgpt_full_memory = df_chatgpt[df_chatgpt["full_memory"] == 1].copy()
df_chatgpt.shape

model7 = smf.ols(
    "originality_index ~ ai_temperature + anti_anchoring_prompt" ,
    data=df_chatgpt_full_memory
).fit()
robust7 = model7.get_robustcov_results(cov_type="HC3")
print(robust7.summary())

cluster7 = model7.get_robustcov_results(
    cov_type="cluster",
    groups=df_chatgpt_full_memory["mystery_word_id"]
)
print(cluster7.summary())



df_chatgpt_full_memory_1_5 = df_chatgpt_full_memory[df_chatgpt_full_memory["ai_temperature"] == 1.5].copy()
df_chatgpt_full_memory_1_5.shape

model8 = smf.ols(
    "originality_index ~  anti_anchoring_prompt" ,
    data=df_chatgpt_full_memory_1_5
).fit()
robust8 = model8.get_robustcov_results(cov_type="HC3")
print(robust8.summary())



"""
cluster8 = model8.get_robustcov_results(
    cov_type="cluster",
    groups=df_chatgpt_full_memory_1_5["mystery_word_id"]
)
print(cluster8.summary())
"""
"""
df_chatgpt_1_5 = df_chatgpt[df_chatgpt["ai_temperature"] == 1.5].copy()
df_chatgpt_1_5.shape

model9 = smf.ols(
    "originality_index ~  full_memory" ,
    data=df_chatgpt_1_5
).fit()
robust9 = model9.get_robustcov_results(cov_type="HC3")
print(robust9.summary())
"""

# Jetzt: LaTeX-Tabelle 
latex_code = ols_to_latex_apa(
    models=[robust5, robust6, robust6_2, robust8, robust7],
    model_names=["(1) Temperature", "(2) + Full Memory","(2a) Full Memory in Temp 2.0", "(3) Prompt-Test", "(4) Prompt-Test"],
    depvar="Originality Index",
    title="ChatGPT settings on Originality",
    label="tab:orig_human_ai",
    notes="Standard Errors are heteroscedasticity robust (HC3) in parentheses. * p<.10, ** p<.05, *** p<.01"
)

print(latex_code)


##################################################Gemini
# OLS just Gemini
##################################################Gemini
df_gemini = df_ai_only[df_ai_only["ai_label"] == "Gemini"].copy()
df_gemini.shape

model_gemini_1 = smf.ols(
    "originality_index ~ ai_temperature + gemini_flash",
    data=df_gemini
).fit()
robust_gemini_1 = model_gemini_1.get_robustcov_results(cov_type="HC3")
print(robust_gemini_1.summary())

cluster_gemini_1 = model_gemini_1.get_robustcov_results(
    cov_type="cluster",
    groups=df_gemini["mystery_word_id"]
)
print(cluster_gemini_1.summary())



df_gemini_1_5 = df_gemini[df_gemini["ai_temperature"] == 1.5].copy()
df_gemini_1_5.shape

model_gemini_2 = smf.ols(
    "originality_index ~ gemini_flash",
    data=df_gemini_1_5
).fit()
robust_gemini_2 = model_gemini_2.get_robustcov_results(cov_type="HC3")
print(robust_gemini_2.summary())

cluster_gemini_2 = model_gemini_2.get_robustcov_results(
    cov_type="cluster",
    groups=df_gemini_1_5["mystery_word_id"]
)
print(cluster_gemini_2.summary())

# Jetzt: LaTeX-Tabelle 
latex_code = ols_to_latex_apa(
    models=[robust_gemini_1, robust_gemini_2],
    model_names=["(1) ", "(2) "],
    depvar="Originality Index",
    title="Gemini-2.5-pro vs -flash",
    label="tab:orig_human_ai",
    notes="Standard Errors are heteroscedasticity robust (HC3) in parentheses.* p<.10, ** p<.05, *** p<.01"
)

print(latex_code)



###########################################################################

#############################################################
# 0) Input assumptions (df_clean)
# - one row = one clue
# - columns: mystery_word_id, human (0/1), human_label (AI/Human),
#            originality_index (0..100), clues (text)
#############################################################

df2 = df_clean.copy()

# Ensure a stable human_label (prefer existing label; fall back to mapping if needed).
if "human_label" not in df2.columns:
    df2["human_label"] = df2["human"].map({0: "AI", 1: "Human"})


#############################################################
# 1) Word-level aggregation helpers
#############################################################

def sd_with_singleton_to_zero(x: pd.Series) -> float:
    """Sample SD for n>1; 0.0 for singleton bundles (SD undefined for n=1)."""
    x = x.dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1))

def clue_at_max_originality(orig_list, clue_list):
    """
    Return the clue corresponding to the first occurrence of the maximum originality.
    Assumes orig_list and clue_list have equal length.
    """
    if not isinstance(orig_list, list) or not isinstance(clue_list, list):
        return np.nan
    if len(orig_list) == 0 or len(clue_list) == 0:
        return np.nan
    # Protect against NaNs inside the list
    clean = [(i, v) for i, v in enumerate(orig_list) if pd.notna(v)]
    if len(clean) == 0:
        return np.nan
    idx_max = max(clean, key=lambda t: t[1])[0]
    return clue_list[idx_max] if idx_max < len(clue_list) else np.nan


#############################################################
# 2) Build word-level dataframe: one row per (mystery_word_id × human_label)
#############################################################

# Lists per bundle (for auditing, later analyses, creative-winner logic).
df_lists = (
    df2.groupby(["mystery_word_id", "human_label"], as_index=False)
      .agg(
          clues_list=("clues", list),
          originality_list=("originality_index", list)
      )
)

# Numeric summaries per bundle.
df_stats = (
    df2.groupby(["mystery_word_id", "human_label"], as_index=False)
      .agg(
          mystery_word=("mystery_word", "first"),
          num_clues=("originality_index", "count"),
          avg_originality=("originality_index", "mean"),
          sd_originality=("originality_index", sd_with_singleton_to_zero),
          min_originality=("originality_index", "min"),
          max_originality=("originality_index", "max"),
      )
)


# Combine to final word-level dataset.
df_wordlevel = df_stats.merge(df_lists, on=["mystery_word_id", "human_label"], how="left")

# Add the clue text corresponding to max originality (creative winner building block).
df_wordlevel["clue_with_max_originality"] = df_wordlevel.apply(
    lambda r: clue_at_max_originality(r["originality_list"], r["clues_list"]),
    axis=1
)


#############################################################
# A) Bundle-level clue frequencies (diagnostics / reporting)
#############################################################

def to_frequency_dict(clue_list):
    """Return a frequency dictionary for a list of clue strings."""
    if not isinstance(clue_list, list) or len(clue_list) == 0:
        return {}
    return dict(Counter(clue_list))

df_wordlevel["clues_frequency"] = df_wordlevel["clues_list"].apply(to_frequency_dict)


#############################################################
# B) Rounded originality lists (optional, useful for bucketed analyses)
#############################################################

def round_list(values, ndigits=0):
    """Round a list of numeric values; returns [] for missing/invalid lists."""
    if not isinstance(values, list) or len(values) == 0:
        return []
    out = []
    for v in values:
        if pd.notna(v):
            out.append(round(float(v), int(ndigits)))
    return out

# Rounded to integers by default (0..100).
df_wordlevel["originality_rounded_list"] = df_wordlevel["originality_list"].apply(lambda x: round_list(x, ndigits=0))


#############################################################
# C) Creative winner dataframe (per mystery word: which condition had the highest max originality)
#############################################################

# Pivot max originality per word (paired only)
wide_max = (
    df_wordlevel
    .pivot(index="mystery_word_id", columns="human_label", values="max_originality")
    .dropna(subset=["AI", "Human"])
)

max_diff = wide_max["AI"] - wide_max["Human"]

winner_summary = pd.DataFrame({
    "mystery_word_id": wide_max.index,
    "max_originality_ai": wide_max["AI"].values,
    "max_originality_human": wide_max["Human"].values,
    "max_diff_ai_minus_human": max_diff.values,
})

winner_summary["winner"] = np.where(
    winner_summary["max_diff_ai_minus_human"] > 0, "AI",
    np.where(winner_summary["max_diff_ai_minus_human"] < 0, "Human", "Tie")
)

# Attach mystery_word text (assumes it exists in df_wordlevel)
word_text = (
    df_wordlevel[["mystery_word_id", "mystery_word"]]
    .drop_duplicates(subset=["mystery_word_id"])
)
winner_summary = winner_summary.merge(word_text, on="mystery_word_id", how="left")

# Build a “winner row” dataframe (for non-ties: take the row of the winning condition)
df_win_candidates = df_wordlevel[
    df_wordlevel["mystery_word_id"].isin(winner_summary["mystery_word_id"])
].copy()

winner_map = winner_summary.set_index("mystery_word_id")["winner"].to_dict()

def is_winner_row(row):
    w = winner_map.get(row["mystery_word_id"])
    return (w in ["AI", "Human"]) and (row["human_label"] == w)

creative_winner = df_win_candidates[df_win_candidates.apply(is_winner_row, axis=1)].copy()

# Keep a clean winner table with the key winner attributes
creative_winner = creative_winner.merge(
    winner_summary[[
        "mystery_word_id",
        "winner",
        "max_originality_ai",
        "max_originality_human",
        "max_diff_ai_minus_human"
        # <-- do NOT merge mystery_word again
    ]],
    on="mystery_word_id",
    how="left"
)


creative_winner = creative_winner[
    [
        "mystery_word_id",
        "mystery_word",
        "winner",
        "human_label",
        "max_originality",
        "clue_with_max_originality",
        "num_clues",
        "avg_originality",
        "sd_originality",
        "max_originality_ai",
        "max_originality_human",
        "max_diff_ai_minus_human",
    ]
].sort_values("max_diff_ai_minus_human", ascending=False)

print("\nCreative winner rows (non-ties only):")
print(creative_winner.head(10))

print("\nCreative winner counts (including ties):")
print(winner_summary["winner"].value_counts())


#############################################################
# D) Word-level differences (AI − Human) for avg_originality and sd_originality
#############################################################

wide_avg = (
    df_wordlevel
    .pivot(index="mystery_word_id", columns="human_label", values="avg_originality")
    .dropna(subset=["AI", "Human"])
)
diff_avg = wide_avg["AI"] - wide_avg["Human"]

wide_sd = (
    df_wordlevel
    .pivot(index="mystery_word_id", columns="human_label", values="sd_originality")
    .dropna(subset=["AI", "Human"])
)
diff_sd = wide_sd["AI"] - wide_sd["Human"]

diff_table = pd.DataFrame({
    "mystery_word_id": wide_avg.index,
    "diff_avg_originality_ai_minus_human": diff_avg.values,
    "diff_sd_originality_ai_minus_human": diff_sd.reindex(wide_avg.index).values,
}).merge(word_text, on="mystery_word_id", how="left")

print("\nDifference table (AI - Human) preview:")
print(diff_table.head(10))



#############################################################
# E) Optional: F-test for variance differences in sd_originality (AI vs Human)
#############################################################

# Note: F-test assumes approximate normality; use as an optional diagnostic.
ai_sd = df_wordlevel.loc[df_wordlevel["human_label"] == "AI", "sd_originality"].dropna().to_numpy()
hu_sd = df_wordlevel.loc[df_wordlevel["human_label"] == "Human", "sd_originality"].dropna().to_numpy()

# Sample variances (ddof=1)
var_ai = np.var(ai_sd, ddof=1)
var_hu = np.var(hu_sd, ddof=1)

# Use ratio of larger to smaller to get a right-tailed F, then convert to two-sided p-value.
if var_ai >= var_hu:
    f_stat = var_ai / var_hu
    dfn, dfd = len(ai_sd) - 1, len(hu_sd) - 1
else:
    f_stat = var_hu / var_ai
    dfn, dfd = len(hu_sd) - 1, len(ai_sd) - 1

p_one_sided = 1 - f.cdf(f_stat, dfn, dfd)
p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)

print("\nOptional F-test (variance comparison) for sd_originality:")
print(f"var(AI) = {var_ai:.6f}, var(Human) = {var_hu:.6f}")
print(f"F = {f_stat:.4f} with df=({dfn}, {dfd}), two-sided p = {p_two_sided:.6g}")








#############################################################
# 3) Descriptive table (word level)
#############################################################

orig_table = (
    df_wordlevel
    .groupby("human_label")
    .agg(
        mean_avg_originality=("avg_originality", "mean"),
        sd_avg_originality=("avg_originality", "std"),
        mean_sd_originality=("sd_originality", "mean"),
        sd_sd_originality=("sd_originality", "std"),
        mean_num_clues=("num_clues", "mean"),
        sd_num_clues=("num_clues", "std"),
        n=("mystery_word_id", "count"),
    )
    .round(3)
)

print("\nWord-level originality descriptives:")
print(orig_table)


#############################################################
# 4) Plots (word level): avg_originality and sd_originality
#############################################################

# KDE: avg_originality
plt.figure(figsize=(10, 5))
ax = sns.kdeplot(
    data=df_wordlevel,
    x="avg_originality",
    hue="human_label",
    common_norm=False,
    fill=True,
    alpha=0.4,
    legend=True
)
plt.title("Distribution of Average Originality (Word level)")
plt.xlabel("Average Originality")
plt.ylabel("Density")
leg = ax.get_legend()
if leg is not None:
    leg.set_title(None)
plt.tight_layout()
plt.show()

# Violin: avg_originality
plt.figure(figsize=(8, 5))
ax = sns.violinplot(
    data=df_wordlevel,
    x="human_label",
    y="avg_originality",
    inner="quartile",
    cut=0
)
plt.title("Average Originality (Word level)")
plt.xlabel("")
plt.ylabel("Average Originality")
plt.tight_layout()
plt.show()

# KDE: sd_originality
plt.figure(figsize=(10, 5))
ax = sns.kdeplot(
    data=df_wordlevel,
    x="sd_originality",
    hue="human_label",
    common_norm=False,
    fill=True,
    alpha=0.4,
    legend=True
)
plt.title("Distribution of Originality SD (Word level)")
plt.xlabel("Standard Deviation of Originality")
plt.ylabel("Density")
leg = ax.get_legend()
if leg is not None:
    leg.set_title(None)
plt.tight_layout()
plt.show()


#############################################################
# 5) Paired t-tests (only where both AI and Human exist for a word)
#############################################################

def paired_ttest_with_ci(word_df, value_col, id_col="mystery_word_id", group_col="human_label",
                         g1="AI", g2="Human", alpha=0.05):
    """
    Paired t-test comparing g1 vs g2 within the same word id.
    Returns t, p, mean_diff (g1-g2), 95% CI, and n pairs.
    """
    wide = (
        word_df.pivot(index=id_col, columns=group_col, values=value_col)
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

    return {
        "t": t_stat,
        "p": p_val,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": n,
        "wide": wide
    }

res_avg = paired_ttest_with_ci(df_wordlevel, "avg_originality")
print("\nPaired t-test (AI - Human, avg_originality):")
print(f"t = {res_avg['t']:.4f}")
print(f"p = {res_avg['p']:.6g}")
print(f"mean difference (AI - Human) = {res_avg['mean_diff']:.3f}")
print(f"95% CI = [{res_avg['ci_low']:.3f}, {res_avg['ci_high']:.3f}]")
print(f"n pairs = {res_avg['n_pairs']}")

res_sd = paired_ttest_with_ci(df_wordlevel, "sd_originality")
print("\nPaired t-test (AI - Human, sd_originality):")
print(f"t = {res_sd['t']:.4f}")
print(f"p = {res_sd['p']:.6g}")
print(f"mean difference (AI - Human) = {res_sd['mean_diff']:.4f}")
print(f"95% CI = [{res_sd['ci_low']:.4f}, {res_sd['ci_high']:.4f}]")
print(f"n pairs = {res_sd['n_pairs']}")


#############################################################
# 6) Creative-winner prep: compare max originality per word (AI vs Human)
#############################################################

wide_max = (
    df_wordlevel.pivot(index="mystery_word_id", columns="human_label", values="max_originality")
               .dropna(subset=["AI", "Human"])
)

max_diff = wide_max["AI"] - wide_max["Human"]

winner_counts = {
    "AI_wins": int((max_diff > 0).sum()),
    "Human_wins": int((max_diff < 0).sum()),
    "Ties": int((max_diff == 0).sum()),
    "n_pairs": int(max_diff.shape[0])
}

print("\nCreative-winner (based on max originality per word):")
print(winner_counts)





########################################################################
########################################################################


"""


df_words = df_clean.copy()

# Label hinzufügen
df_words["human_label"] = df_words["human"].map({0: "AI", 1: "Human"})

# Hilfsfunktion: Häufigkeiten der Clues
def freq_dict(series):
    return series.value_counts().to_dict()

# Aggregation über alle Clues pro Wort *und* Human/Ai
word_level = (
    df_words
    .groupby(["mystery_word_id", "mystery_word", "human", "human_label"], as_index=False)
    .agg(
        clues_list=("cleaned_clue", list),
        originality_list=("originality_index", list),
        originality_rounded_list=("originality_index_rounded", list),
        clues_frequency=("cleaned_clue", freq_dict),
        min_originality=("originality_index", "min"),
        max_originality=("originality_index", "max"),
        avg_originality=("originality_index", "mean"),
        sd_originality=("originality_index", "std")
    )
)

# Funktion: Clue mit der höchsten Originalität
def clue_with_max_originality(row):
    clues = row["clues_list"]
    scores = row["originality_list"]
    if len(scores) == 0:
        return None
    idx_max = max(range(len(scores)), key=lambda i: scores[i])
    return clues[idx_max]

word_level["clue_with_max_originality"] = word_level.apply(
    clue_with_max_originality, axis=1
)

word_level.head()


#####################################################


# Für jedes mystery_word_id die Zeile mit der höchsten max_originality auswählen
creative_winner = (
    word_level
    .sort_values(["mystery_word_id", "max_originality"], ascending=[True, False])
    .groupby("mystery_word_id", as_index=False)
    .first()
)

creative_winner = creative_winner[
    ["mystery_word_id", "mystery_word", "human_label", 
     "max_originality", "clue_with_max_originality"]
]

creative_winner.head()

creative_winner["human_label"].value_counts()

ai_winner_words = creative_winner[creative_winner["human_label"] == "AI"]
ai_winner_words



winner_ai_matches = []

for idx, row in ai_winner_words.iterrows():
    w = row["mystery_word"]                      # Wort
    clue = row["clue_with_max_originality"]     # Gewinner-Clue
    
    # Suche in df_clean
    match = df_clean[
        (df_clean["mystery_word"] == w) &
        (df_clean["cleaned_clue"] == clue)
    ]
    
    if len(match) > 0:
        winner_ai_matches.append(match)

# Alle gefundenen Zeilen zusammenführen
winner_ai_matches_df = pd.concat(winner_ai_matches, ignore_index=True)

winner_ai_matches_df


#############

# Tabelle von AI–Human als breite Tabelle
wide = word_level.pivot(
    index="mystery_word_id",
    columns="human_label",
    values="avg_originality"
)

# Optional: Spalten schöner machen
wide = wide.rename(columns={"Human": "human_avg", "AI": "ai_avg"})

# Differenz berechnen (positiv = AI kreativer)
wide["difference_ai_minus_human"] = wide["ai_avg"] - wide["human_avg"]

wide.head()




ttest_result = ttest_rel(wide["ai_avg"], wide["human_avg"])
print(ttest_result)

# paired differences
diff = wide["ai_avg"] - wide["human_avg"]
n = len(diff)

# t-test
t_stat, p_value = ttest_rel(wide["ai_avg"], wide["human_avg"])

# standard error
se = diff.std(ddof=1) / np.sqrt(n)

# critical t-value (two-sided, 95%)
t_crit = t.ppf(0.975, df=n-1)

# CI for the mean difference
mean_diff = diff.mean()
ci_lower = mean_diff - t_crit * se
ci_upper = mean_diff + t_crit * se

# Summary Table
ttest_table = {
    "Test": ["Paired t-test"],
    "t-statistic": [t_stat],
    "p-value": [p_value],
    "Mean Difference (AI − Human)": [mean_diff],
    "95% CI Lower": [ci_lower],
    "95% CI Upper": [ci_upper],
    "N (words)": [n]
}

ttest_df = pd.DataFrame(ttest_table)
ttest_df


####### Plot Difference in Averageg Originality

wide = wide.reset_index()              # falls nötig
wide = wide.set_index("mystery_word_id")

plt.figure(figsize=(14,5))
ax = wide["difference_ai_minus_human"].plot(kind="bar")

plt.axhline(0, color="black")
plt.title("AI - Human Average Originality Difference per Word")
plt.ylabel("Difference")
plt.xlabel("Mystery Word")
# echte Word-IDs
word_ids = wide.index.to_list()

# jede 50. Word-ID finden (50, 100, 150, ...)
tick_labels = [wid for wid in word_ids if wid % 50 == 0]

# deren Positionen im Barplot ermitteln
tick_positions = [word_ids.index(wid) for wid in tick_labels]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=70)

plt.show()

################## Avergag Originality per word ai vs human
overall = (
    word_level
    .groupby("human_label")["avg_originality"]
    .describe()
)
overall
"""
"""
plt.figure(figsize=(10,6))
sns.kdeplot(
    data=word_level,
    x="avg_originality",
    hue="human_label",
    common_norm=False,
    fill="True",
    alpha=0.25
)

plt.title("Distrubution of Average Originality per Word")
plt.xlabel("Average Originality (mean per word)")
plt.show()
"""
"""

ax = sns.kdeplot(
    data=word_level,
    x="avg_originality",
    hue="human_label",
    fill=True,
    alpha=0.4
)

# Legendentitel entfernen (aber Legende behalten!)
legend = ax.get_legend()
legend.set_title("")

plt.title("Distrubution of Average Originality per Word")
plt.xlabel("Average Originality (mean per word)")
plt.ylabel("Density")

plt.tight_layout()
plt.show()



"""
"""
plt.figure(figsize=(8,6))
sns.violinplot(
    data=word_level,
    x="human_label",
    y="avg_originality"
)
plt.title("Violinplot of Average Originality, word-level")
plt.xlabel("")
plt.ylabel("Average Originality")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(
    data=word_level,
    x="human_label",
    y="avg_originality"
)
plt.title("Boxplot der durchschnittlichen Originalität pro Wort")
plt.show()
"""
"""
##################  std. dev Originality per word ai vs human
word_level.groupby("human_label")["sd_originality"].describe()
desc_df = word_level.groupby("human_label")["sd_originality"].describe()


plt.figure(figsize=(10,6))
sns.kdeplot(
    data=word_level,
    x="sd_originality",
    hue="human_label",
    common_norm=False,
    fill=True,
    alpha=0.4
)
plt.title("Distribution of Standard Deviation of Originality, word-level (Human vs AI)")
plt.xlabel("Standard Deviation of Originality")
plt.show()




######Varianz F-test
ai_sd = word_level[word_level["human_label"] == "AI"]["sd_originality"].dropna()
human_sd = word_level[word_level["human_label"] == "Human"]["sd_originality"].dropna()

var_ai = np.var(ai_sd, ddof=1)
var_human = np.var(human_sd, ddof=1)

F = var_ai / var_human
F

###################TTEst Std abweichung

wide_sd = word_level.pivot(
    index="mystery_word_id",
    columns="human_label",
    values="sd_originality"
)

# Fälle entfernen, wo SD fehlt
wide_sd = wide_sd.dropna()

# ----------------------------
# 2. Paired t-test
# ----------------------------

t_stat, p_val = ttest_rel(wide_sd["AI"], wide_sd["Human"])

print("Paired t-test (SD: AI vs Human)")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# ----------------------------
# 3. 95% Confidence Interval
# ----------------------------

diff = wide_sd["AI"] - wide_sd["Human"]
n = len(diff)
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)

# t-critical value
t_crit = t.ppf(1 - 0.05/2, df=n-1)

# Margin of Error
moe = t_crit * std_diff / np.sqrt(n)

ci_lower = mean_diff - moe
ci_upper = mean_diff + moe

print("\n95% Confidence Interval for mean difference (AI - Human):")
print(f"Mean difference: {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Sample size (paired): n = {n}")





# Variances
var_ai = np.var(ai_sd, ddof=1)
var_human = np.var(human_sd, ddof=1)

# F statistic (AI variance / Human variance)
F_stat = var_ai / var_human

df1 = len(ai_sd) - 1
df2 = len(human_sd) - 1

# p-value (two-sided)
p_right = 1 - f.cdf(F_stat, df1, df2)
p_left  = f.cdf(F_stat, df1, df2)
p_two_sided = 2 * min(p_right, p_left)

# 95% CI
alpha = 0.05
F_lower = f.ppf(alpha/2, df1, df2)
F_upper = f.ppf(1 - alpha/2, df1, df2)

ci_lower = F_stat / F_upper
ci_upper = F_stat / F_lower

# Summary Table
ftest_table = {
    "Test": ["F-test (AI variance / Human variance)"],
    "F-statistic": [F_stat],
    "p-value": [p_two_sided],
    "Variance Ratio (AI/Human)": [F_stat],
    "95% CI Lower": [ci_lower],
    "95% CI Upper": [ci_upper],
    "df1": [df1],
    "df2": [df2]
}

ftest_df = pd.DataFrame(ftest_table)
ftest_df

"""
##############################################################



# Create sets of AI and Human clues per word
grouped = df_clean.groupby("mystery_word_id", group_keys=False)

ai_human_sets = grouped.apply(
    lambda g: pd.Series({
        "mystery_word": g["mystery_word"].iloc[0],
        "ai_clues": set(g.loc[g["human"] == 0, "cleaned_clue"]),
        "human_clues": set(g.loc[g["human"] == 1, "cleaned_clue"]),
    }),
    include_groups=False
).reset_index()


# AI-only clues
ai_human_sets["ai_only_clues"] = (
    ai_human_sets["ai_clues"] - ai_human_sets["human_clues"]
)

# Count AI-only clues
ai_human_sets["ai_only_count"] = ai_human_sets["ai_only_clues"].apply(len)

# Total AI clues count (unique)
ai_human_sets["ai_total_unique"] = ai_human_sets["ai_clues"].apply(len)

# % of AI clues that are unique
ai_human_sets["ai_unique_percent"] = (
    ai_human_sets["ai_only_count"] / ai_human_sets["ai_total_unique"] * 100
)

# Ranking: words where AI has most unique clues
ai_ranked = ai_human_sets.sort_values("ai_only_count", ascending=False)

ai_ranked[
    ["mystery_word_id", "mystery_word", "ai_only_count", "ai_unique_percent", "ai_only_clues"]
]


#############
# 1. Total number of unique AI-only clues across ALL words
total_ai_only_clues = ai_human_sets["ai_only_clues"].apply(len).sum()

# 2. Average number of AI-only clues per word
avg_ai_only_per_word = ai_human_sets["ai_only_count"].mean()

# 3. Average % of unique AI clues per word
avg_ai_unique_percent = ai_human_sets["ai_unique_percent"].mean()

# 4. Collect all unique AI-only clues across all words
all_ai_only_clues_flat = set().union(*ai_human_sets["ai_only_clues"])

# Build summary table
summary = pd.DataFrame({
    "Metric": [
        "Total AI-only clues",
        "Average AI-only clues per word",
        "Average % unique AI clues per word",
        "Total unique AI-only clue types"
    ],
    "Value": [
        total_ai_only_clues,
        avg_ai_only_per_word,
        avg_ai_unique_percent,
        len(all_ai_only_clues_flat)
    ]
})

summary


total_ai_clues = df_clean[df_clean["human"] == 0].shape[0]
total_ai_clues
print("Total number of AI clues:", total_ai_clues)

perc_unique_ai = (total_ai_only_clues / total_ai_clues) * 100

perc_unique_ai


# 1. Alle AI-only clues (aus deiner bestehenden Tabelle)
# 2. Filtere AI-only rows aus df_clean
ai_only_rows = df_clean[
    (df_clean["human"] == 0) &
    (df_clean["cleaned_clue"].isin(all_ai_only_clues_flat))
]

# 3. Originality-Werte aller AI-only clues
ai_only_originality = ai_only_rows["originality_index"]

# 4. Mean der AI-only originality
mean_ai_only_originality = ai_only_originality.mean()

print("Mean originality of AI-only clues:", mean_ai_only_originality)



###########################################################
#AI-only clues and metrics

# ----------------------------
# 0) Build per-mystery-word clue sets (AI vs Human)
# ----------------------------
grouped = df_clean.groupby("mystery_word_id", group_keys=False)

ai_human_sets = (
    grouped.apply(
        lambda g: pd.Series({
            "mystery_word": g["mystery_word"].iloc[0],
            "ai_clues": set(g.loc[g["human"] == 0, "cleaned_clue"]),
            "human_clues": set(g.loc[g["human"] == 1, "cleaned_clue"]),
        }),
        include_groups=False
    )
    .reset_index()
)

# ----------------------------
# 1) Compute AI-only clues per mystery word (set difference)
# ----------------------------
ai_human_sets["ai_only_clues"] = ai_human_sets["ai_clues"] - ai_human_sets["human_clues"]

# Counts per mystery word
ai_human_sets["ai_only_count"] = ai_human_sets["ai_only_clues"].apply(len)
ai_human_sets["ai_total_unique"] = ai_human_sets["ai_clues"].apply(len)

# Percent of AI clues that are AI-only (per mystery word)
# Division-by-zero protection for cases with no AI clues
ai_human_sets["ai_unique_percent"] = (
    ai_human_sets["ai_only_count"] / ai_human_sets["ai_total_unique"].replace({0: pd.NA}) * 100
)

# ----------------------------
# 2) Global metrics requested
# ----------------------------

# Total AI-only clues across all mystery words (duplicates across words counted)
total_ai_only_clues = ai_human_sets["ai_only_count"].sum()

# Number of mystery words that have at least one AI-only clue
num_words_with_ai_only = (ai_human_sets["ai_only_count"] > 0).sum()

# Mean percent of AI-only clues among AI clues (per mystery word)
avg_ai_unique_percent = ai_human_sets["ai_unique_percent"].mean(skipna=True)

# ----------------------------
# 3) AI-only clue table with originality_index (row-level, per word)
# ----------------------------

# Expand AI-only clue sets into one row per (mystery_word_id, cleaned_clue)
ai_only_long = (
    ai_human_sets.loc[:, ["mystery_word_id", "mystery_word", "ai_only_clues"]]
    .explode("ai_only_clues")
    .rename(columns={"ai_only_clues": "cleaned_clue"})
)

# Remove rows created by exploding empty sets
ai_only_long = ai_only_long.dropna(subset=["cleaned_clue"])

# Map originality_index from the underlying AI rows; enforce one value per (word, clue)
orig_map = (
    df_clean.loc[df_clean["human"] == 0, ["mystery_word_id", "cleaned_clue", "originality_index"]]
    .drop_duplicates(subset=["mystery_word_id", "cleaned_clue"])
)

# Attach originality_index to each AI-only (word, clue) pair
ai_only_with_orig = ai_only_long.merge(
    orig_map,
    on=["mystery_word_id", "cleaned_clue"],
    how="left"
)

# Global mean originality_index across all AI-only clues (duplicates across words counted)
mean_ai_only_originality_global = ai_only_with_orig["originality_index"].mean()

# ----------------------------
# 4) Top/Bottom 25 AI-only clues by originality_index
# ----------------------------
cols_out = ["mystery_word_id", "mystery_word", "cleaned_clue", "originality_index"]

top25_ai_only = (
    ai_only_with_orig.sort_values("originality_index", ascending=False)
    .loc[:, cols_out]
    .head(25)
)

bottom25_ai_only = (
    ai_only_with_orig.sort_values("originality_index", ascending=True)
    .loc[:, cols_out]
    .head(25)
)

# ----------------------------
# 5) Summary table (all requested metrics)
# ----------------------------
summary = pd.DataFrame({
    "Metric": [
        "Total AI-only clues (summed across mystery words; duplicates across words counted)",
        "Number of mystery words with at least 1 AI-only clue",
        "Mean % of AI-only clues among AI clues (per mystery word)",
        "Global mean originality_index of all AI-only clues"
    ],
    "Value": [
        total_ai_only_clues,
        num_words_with_ai_only,
        avg_ai_unique_percent,
        mean_ai_only_originality_global
    ]
})

# Outputs
summary, ai_human_sets, ai_only_with_orig, top25_ai_only, bottom25_ai_only




########################### Alle AI only clues in latex tabelle
def format_clue_set(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (set, list, tuple)):
        items = [str(i).strip().strip("'\"") for i in x if str(i).strip()]
    else:
        return str(x).replace("{", "").replace("}", "").replace("'", "").replace('"', "")
    items = sorted(set(items), key=lambda z: z.lower())
    return ", ".join(items)

def df_rows_to_latex(d: pd.DataFrame) -> str:
    """
    Gibt nur die Datenzeilen als LaTeX zurück:
    z.B. "2 & Alarm & emergency, urgency \\\\"
    """
    if d.empty:
        return r"\multicolumn{3}{l}{\emph{(keine Einträge)}} \\"
    lines = []
    for _, r in d.iterrows():
        lines.append(f"{int(r['mystery_word_id'])} & {r['mystery_word']} & {r['ai_only_clues']} \\\\")
    return "\n".join(lines)


df_latex = (
    ai_human_sets.loc[
        ai_human_sets["ai_only_count"] > 0,
        ["mystery_word_id", "mystery_word", "ai_only_clues"]
    ]
    .sort_values("mystery_word_id", ascending=True)
    .reset_index(drop=True)
    .copy()
)

df_latex["ai_only_clues"] = df_latex["ai_only_clues"].apply(format_clue_set)

ROWS_PER_COLUMN = 50  # <-- ggf. anpassen

blocks = []
page = 1

for start in range(0, len(df_latex), 2 * ROWS_PER_COLUMN):
    left = df_latex.iloc[start : start + ROWS_PER_COLUMN]
    right = df_latex.iloc[start + ROWS_PER_COLUMN : start + 2 * ROWS_PER_COLUMN]

    left_rows = df_rows_to_latex(left)
    right_rows = df_rows_to_latex(right)

    block = rf"""
% =========================
% AI-only clues: Seite {page}
% =========================
\clearpage
{{\small
\setlength{{\tabcolsep}}{{2pt}}
\renewcommand{{\arraystretch}}{{0.85}}
\begin{{spacing}}{{1.0}}

\begin{{paracol}}{{2}}

% -------- LINKE SPALTE --------
\begin{{center}}
\begin{{tabular}}{{r l P{{4.8cm}}}}
\toprule
id & mystery word & ai-only clues\\
\midrule
{left_rows}
\bottomrule
\end{{tabular}}
\end{{center}}

\switchcolumn

% -------- RECHTE SPALTE --------
\begin{{center}}
\begin{{tabular}}{{r l P{{4.8cm}}}}
\toprule
id & mystery word & ai-only clues\\
\midrule
{right_rows}
\bottomrule
\end{{tabular}}
\end{{center}}

\end{{paracol}}

\end{{spacing}}
}}
"""
    blocks.append(block)
    page += 1

latex_all_pages = "\n".join(blocks)
print(latex_all_pages)



##########################################################
## OLS originality ~ mystery_word id
##########################################################


robust_test_trend_ai = smf.ols(
    "originality_index ~ mystery_word_id",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3"
)
print(robust_test_trend_ai.summary())


robust_test_trend_chatgpt = smf.ols(
    "originality_index ~ mystery_word_id * full_memory",
    data=df_chatgpt
).fit(
    cov_type="HC3"
)
print(robust_test_trend_chatgpt.summary())


robust_test_trend_ai_2 = smf.ols(
    "originality_index ~ -1 + C(ai_label)+ mystery_word_id ",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3"
)
print(robust_test_trend_ai_2.summary())

robust_test_trend_ai_3 = smf.ols(
    "originality_index ~ -1 + C(ai_label) * mystery_word_id ",
    data=df_ai_full_memory_only
).fit(
    cov_type="HC3"
)
print(robust_test_trend_ai_3.summary())


# Jetzt: LaTeX-Tabelle 
latex_code = ols_to_latex_apa(
    models=[robust_test_trend_ai],
    model_names=["(1) "],
    depvar="Originality Index",
    title="Trend Originality ",
    label="tab:trend_orig",
    notes="Standard Errors are heteroscedasticity robust (HC3) in parentheses.* p<.10, ** p<.05, *** p<.01"
)

print(latex_code)
