# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:30:36 2025

@author: serge
"""
import pandas as pd
import ast
from tqdm import tqdm

# tqdm für pandas apply aktivieren
tqdm.pandas()

# 1. CSV einlesen
df = pd.read_csv(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv")

# 2. Sichere Parser-Funktion
def parse_clues(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        print("Problem beim Parsen von:", x)
        return []

# 3. Fortschrittsbalken beim Parsen
df["Clues"] = df["Clues"].progress_apply(parse_clues)

# 4. IDs vergeben
df["mystery_word_id"] = df.index + 1

# 5. Explodieren
df = df[["mystery_word_id", "Mystery Word", "Clues"]]
df = df.explode("Clues")

# 6. Leere Clues entfernen
df = df[df["Clues"].notna()]

# 7. Spalten umbenennen
df = df.rename(columns={
    "Mystery Word": "mystery_word",
    "Clues": "clues"
})

# 8. Speichern
output_path = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\clues_exploded.csv"
df.to_csv(output_path, index=False)

print("Fertig! Datei gespeichert unter:", output_path)

