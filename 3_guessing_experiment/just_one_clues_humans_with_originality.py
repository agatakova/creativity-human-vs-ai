# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:59:52 2025

@author: serge
"""

# ============================================================
# ======================= KONFIGURATION =======================
# ============================================================

import pandas as pd
import numpy as np
import fasttext
from scipy.spatial.distance import cosine
from collections import Counter
import string
import ast
import random
from typing import List, Any

# ----- Dateien & Pfade -----
INPUT_CSV = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv"
OUTPUT_CSV = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_clues_random_human_sample.csv"

# FastText English Model
FASTTEXT_MODEL_PATH = r"C:\Users\serge\Documents\Thesis\Data Seafile\Code\cc.en.300.bin"

# ----- CSV-Formate -----

INPUT_SEP = ','      # Trennzeichen Input-CSV (z.B. ',' oder ';')
OUTPUT_SEP = ','     # Trennzeichen Output-CSV
ENCODING = "utf-8"   # Oder z.B. "latin1"
NROWS = None         # z.B. 100 für Test, oder None für alle Zeilen

# ----- Sampling-Einstellungen -----

RANDOM_SEED = 42         # Für reproduzierbare Ergebnisse
NUM_RANDOM_CLUES = 4     # Wie viele Clues pro Mystery Word ziehen

# Sampling-Variante:
# False = ohne Zurücklegen (standard) -> Duplikate nur, wenn im Original schon doppelt
# True  = mit Zurücklegen -> Duplikate können auch aus einmaligem Vorkommen entstehen
SAMPLE_WITH_REPLACEMENT = False

# ----- Spaltennamen -----

COL_MYSTERY = "Mystery Word"        # Mystery-Word-Spalte im Input
CLUE_COLUMNS = ["Clues"]            # Spalten mit Clue-Listen/Dicts
RANDOM_CLUES_COLUMN = "RandomHumanClues"   # Name für die gezogene Clue-Liste

LANGUAGE = 'en'   # Sprache für FastText-Modell


# ============================================================
# ==================== MODEL-WRAPPER =========================
# ============================================================

class Model:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(Model, cls).__new__(cls)
            cls.instance.models = {}
        return cls.instance

    def __init__(self):
        if not hasattr(self, 'models'):
            self.models = {}
        if not hasattr(self, 'vector_cache'):
            # Cache: (language, word) -> vector
            self.vector_cache = {}

    def _load_model(self, language, model_path):
        if language not in self.models:
            print(f'Loading FastText model "{language}" from {model_path}...')
            self.models[language] = fasttext.load_model(model_path)
            print("Model loaded successfully.")

    def get_vector(self, word, language, model_path):
        key = (language, word)
        if key in self.vector_cache:
            return self.vector_cache[key]

        self._load_model(language, model_path)
        vec = self.models[language].get_word_vector(word)
        self.vector_cache[key] = vec
        return vec

    def distance(self, word1: str, word2: str, language: str, model_path: str):
        """
        Cosine-Distanz * 100, integer-gerundet.
        Beide Wörter werden lowercase behandelt.
        """
        word1 = str(word1).lower()
        word2 = str(word2).lower()

        vector1 = self.get_vector(word1, language, model_path)
        vector2 = self.get_vector(word2, language, model_path)

        if vector1 is not None and vector2 is not None:
            return round(cosine(vector1, vector2) * 100)
        else:
            return None


# ============================================================
# ===================== HILFSFUNKTIONEN ======================
# ============================================================

def normalize_word(w: str) -> str:
    """Trim, Satzzeichen außen weg, lowercase."""
    return str(w).strip().strip(string.punctuation).lower()


def parse_clue_cell(cell: Any) -> List[str]:
    """
    Parst eine Clue-Zelle, z.B.:
      - "['inflection', 'twang', 'speak']"
      - "{'dialect': 15, 'foreign': 10, ...}"
    Im Dict-Fall werden nur die Keys zurückgegeben.
    """
    if pd.isna(cell):
        return []

    if isinstance(cell, (list, tuple, set, dict)):
        obj = cell
    else:
        text = str(cell).strip()
        if not text:
            return []
        try:
            obj = ast.literal_eval(text)
        except Exception:
            return []

    if isinstance(obj, dict):
        clues = list(obj.keys())
    elif isinstance(obj, (list, tuple, set)):
        clues = list(obj)
    else:
        return []

    return [str(c) for c in clues if c is not None]


def sample_clues(all_clues: List[str]) -> List[str]:
    """
    Wählt zufällig Clues aus all_clues:
    - SAMPLE_WITH_REPLACEMENT steuert mit/ohne Zurücklegen
    - NUM_RANDOM_CLUES gibt die Zielanzahl an
    - Wenn ohne Zurücklegen und zu wenige Clues vorhanden sind:
        -> nimm alle vorhandenen.
    """
    if not all_clues:
        return []

    if SAMPLE_WITH_REPLACEMENT:
        return random.choices(all_clues, k=NUM_RANDOM_CLUES)
    else:
        if len(all_clues) <= NUM_RANDOM_CLUES:
            return list(all_clues)
        else:
            return random.sample(all_clues, NUM_RANDOM_CLUES)


# ============================================================
# ===================== ZEILENVERARBEITUNG ===================
# ============================================================

def process_row(row, model: Model):
    # Mystery Word
    mystery_raw = str(row[COL_MYSTERY]).strip()
    mystery_word = mystery_raw.lower()

    # Alle Clues aus allen CLUE_COLUMNS einsammeln (roh)
    all_clues_raw: List[str] = []
    for col in CLUE_COLUMNS:
        if col in row:
            all_clues_raw.extend(parse_clue_cell(row[col]))

    # Zufällig Clues sampeln (mit Seed, mit/ohne Zurücklegen)
    sampled_clues_raw = sample_clues(all_clues_raw)

    # Normalisierte Version der gezogenen Clues (für Frequency & Originality)
    clues_clean = []
    for c in sampled_clues_raw:
        w = normalize_word(c)
        if w:
            clues_clean.append(w)

    # ---- RandomHumanClues (im gewünschten Listenformat, ungefiltert) ----
    random_clues_str = repr(sampled_clues_raw)

    # ---- Frequenzen (auf normalisierten gezogenen Clues) ----
    freq_dict = dict(Counter(clues_clean))
    freq_str = repr(freq_dict)

    # ---- UniqueClues: Clues mit Frequenz == 1 ----
    unique_clues = [word for word, count in freq_dict.items() if count == 1]
    unique_clues_str = repr(unique_clues)
    num_unique_clues = len(unique_clues)

    # ---- Originalität für ALLE gezogenen Clues (normalisiert) ----
    originality_all = []
    for clue in clues_clean:
        dist = model.distance(clue, mystery_word, LANGUAGE, FASTTEXT_MODEL_PATH)
        if dist is not None:
            originality_all.append(dist)

    originality_all_str = repr(originality_all)

    if len(originality_all) > 0:
        min_orig_all = int(np.min(originality_all))
        max_orig_all = int(np.max(originality_all))
        avg_orig_all = float(np.mean(originality_all))
        std_orig_all = float(np.std(originality_all, ddof=0))
    else:
        min_orig_all = max_orig_all = avg_orig_all = std_orig_all = np.nan

    # ---- Originalität NUR für UniqueClues ----
    originality_unique = []
    for clue in unique_clues:
        dist = model.distance(clue, mystery_word, LANGUAGE, FASTTEXT_MODEL_PATH)
        if dist is not None:
            originality_unique.append(dist)

    originality_unique_str = repr(originality_unique)

    if len(originality_unique) > 0:
        min_orig_uniq = int(np.min(originality_unique))
        max_orig_uniq = int(np.max(originality_unique))
        avg_orig_uniq = float(np.mean(originality_unique))
        std_orig_uniq = float(np.std(originality_unique, ddof=0))
    else:
        min_orig_uniq = max_orig_uniq = avg_orig_uniq = std_orig_uniq = np.nan

    # Spaltenreihenfolge wie besprochen
    return pd.Series({
        'Mystery Word': mystery_raw,
        RANDOM_CLUES_COLUMN: random_clues_str,
        'OriginalityScores_All': originality_all_str,
        'MinOriginality_All': min_orig_all,
        'MaxOriginality_All': max_orig_all,
        'AverageOriginality_All': avg_orig_all,
        'StdOriginality_All': std_orig_all,
        'ClueFrequency': freq_str,
        'UniqueClues': unique_clues_str,
        'NumUniqueClues': num_unique_clues,
        'OriginalityScores_Unique': originality_unique_str,
        'MinOriginality_Unique': min_orig_uniq,
        'MaxOriginality_Unique': max_orig_uniq,
        'AverageOriginality_Unique': avg_orig_uniq,
        'StdOriginality_Unique': std_orig_uniq
    })


# ============================================================
# ========================= MAIN =============================
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    print("Lese Input-CSV...")
    df = pd.read_csv(
        INPUT_CSV,
        sep=INPUT_SEP,
        encoding=ENCODING,
        nrows=NROWS
    )

    missing = [c for c in [COL_MYSTERY] + CLUE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in der Input-CSV: {missing}")

    model = Model()

    print("Sample Clues & berechne Originalität...")
    result_df = df.apply(lambda row: process_row(row, model), axis=1)

    print("Schreibe Output-CSV...")
    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        sep=OUTPUT_SEP,
        encoding=ENCODING
    )

    print(f"Fertig! Datei gespeichert unter:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
