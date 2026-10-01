# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:58:56 2025

@author: serge
"""
import pandas as pd
import numpy as np
import ast

# =========================
# 1) Pfade hier anpassen
# =========================
path_df1 = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses.csv"
path_df2 = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses_unique_clues.csv"
path_df3 = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses_unique_clues_humans.csv"
# =========================
# 2) Hilfsfunktion zum Laden & Aufbereiten
# =========================

def load_and_prepare_guess_data(path):
    # wichtig: keep_default_na=False
    df = pd.read_csv(path, keep_default_na=False)

    # clues parsen
    def parse_clues(x):
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() in ["nan"]:
                return []
            try:
                return ast.literal_eval(x)
            except (SyntaxError, ValueError):
                return []
        elif isinstance(x, (set, list, tuple)):
            return x
        else:
            return []

    df["clues"] = df["clues"].apply(parse_clues)
    df["num_clues"] = df["clues"].apply(lambda c: len(c) if isinstance(c, (set, list, tuple)) else 0)

    # attempts_used als String behandeln
    df["attempts_used"] = df["attempts_used"].astype(str).str.strip()

    # "None" -> "0", "invalid_no_clues" -> NaN
    df["attempts_used"] = df["attempts_used"].replace({
        "None": "0",
        "none": "0",
        "invalid_no_clues": np.nan,
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
    })

    # jetzt in Zahlen umwandeln
    df["attempts_used"] = pd.to_numeric(df["attempts_used"], errors="coerce")

    return df


# =========================
# 3) Datensätze laden
# =========================
df1 = load_and_prepare_guess_data(path_df1)
df2 = load_and_prepare_guess_data(path_df2)
df3 = load_and_prepare_guess_data(path_df3)

# kurz checken
print(df1.head())
print(df1[["mystery_word", "clues", "num_clues", "attempts_used"]].head())



df1["human"] = 0; df1["unique"] = 0
df2["human"] = 0; df2["unique"] = 1
df3["human"] = 1; df3["unique"] = 1

df_all = pd.concat([df1, df2, df3], ignore_index=True)

print(df_all["attempts_used"].value_counts(dropna=False))
print(df_all[["mystery_word", "num_clues", "attempts_used", "human", "unique"]].head())

output_path = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Outpu\guesses_combined_cleaned.csv"
df_all.to_csv(output_path, index=False, encoding="utf-8")

print("CSV gespeichert unter:", output_path)
