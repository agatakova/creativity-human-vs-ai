# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 18:51:21 2025

@author: serge
"""

# ============================================================
# ======================= KONFIGURATION =======================
# ============================================================

# Haupt-CSV mit 4 Clue-Spalten

EXCEL_PATH = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\saved\just_one_clues_hightemp_allmodels.csv"
OUTPUT_CSV = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\saved\just_one_clues_hightemp_allmodels_originality.csv"

# FastText English Model
FASTTEXT_MODEL_PATH = r"C:\Users\serge\Documents\Thesis\Data Seafile\Code\cc.en.300.bin"

# Zweite CSV mit bereits existierenden Clues in Listenform (z.B. "['inflection', 'twang', ...]")
OTHER_CSV_PATH = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv"

# Spaltennamen deiner 4 Clue-Quellen
CLUE_COLUMNS = [
    'ClueTemperature_1_5_ChatGPT',
    'ClueTemperature_1_5_Deep_Seek',
    'ClueTemperature_1_0_Claude',
    'ClueTemperature_1_5_pro_Gemini'
]

LANGUAGE = 'en'

# Name der Spalte in der anderen CSV, in der die Clue-Liste steht (3. Spalte bei dir)
# Beispielformat: "['inflection', 'twang', 'speak', 'dialect', 'voice', 'scouse']"
OTHER_CLUE_COLUMN = 'Clues'  # <-- 

# ============================================================
# ================ AB HIER NICHTS MEHR ÄNDERN =================
# ============================================================

import pandas as pd
import numpy as np
import fasttext
from scipy.spatial.distance import cosine
from collections import Counter
import string
import ast


# --------------------------------------------------
# FastText-Model Wrapper mit Vector-Cache
# --------------------------------------------------

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
        """
        Holt den Vektor aus dem Cache oder aus dem FastText-Modell.
        """
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


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------

def normalize_word(w: str) -> str:
    """Trim, Satzzeichen außen weg, lowercase."""
    return w.strip().strip(string.punctuation).lower()


def parse_clue_list_string(s):
    """
    Nimmt einen String wie "['inflection', 'twang']"
    -> Liste von normalisierten Strings.
    """
    if not isinstance(s, str) or not s.strip():
        return []

    try:
        raw_list = ast.literal_eval(s)
    except Exception:
        # Fallback, falls das Format mal nicht ganz sauber ist
        raw_list = [part.strip(" []'\"") for part in s.split(',')]

    cleaned = []
    for item in raw_list:
        if isinstance(item, str):
            w = normalize_word(item)
            if w:
                cleaned.append(w)
    return cleaned


def build_other_clues_map(other_df, other_clue_column):
    """
    Baut ein Mapping:
        key: mystery_word_lower
        value: set(normalisierter_clues_aus_anderer_datei)
    """
    other_map = {}
    if 'Mystery Word' not in other_df.columns:
        raise ValueError("In der anderen CSV wurde keine Spalte 'Mystery Word' gefunden.")

    for _, row in other_df.iterrows():
        mw_raw = str(row['Mystery Word']).strip()
        key = mw_raw.lower()

        clues_str = row.get(other_clue_column, "")
        clues_list = parse_clue_list_string(clues_str)
        other_map[key] = set(clues_list)

    return other_map


# --------------------------------------------------
# Verarbeitung pro Zeile
# --------------------------------------------------

def process_row(row, model, clue_columns, language, model_path, other_clues_map):
    # Mystery Word
    mystery_raw = str(row['Mystery Word']).strip()
    mystery_word = mystery_raw.lower()

    # Eigene Clues normalisieren
    clues_clean = []
    for col in clue_columns:
        val = row.get(col, "")
        if isinstance(val, str):
            w = normalize_word(val)
            if w:
                clues_clean.append(w)

    # Alle Clues als String (für CSV)
    clues_str = repr(clues_clean)

    # Häufigkeiten
    freq_dict = dict(Counter(clues_clean))
    freq_str = repr(freq_dict)

    # UniqueClues (Frequenz == 1)
    unique_clues = [word for word, count in freq_dict.items() if count == 1]
    unique_clues_str = repr(unique_clues)

    # Clues aus anderer Datei zu diesem Mystery Word
    other_clues_set = other_clues_map.get(mystery_word, set())

    # Eigene Clues, die NICHT in anderer Datei vorkommen
    clues_not_in_other = [c for c in clues_clean if c not in other_clues_set]
    clues_not_in_other_str = repr(clues_not_in_other)

    # Originalität für ALLE eigenen Clues
    originality_scores = []
    for clue in clues_clean:
        dist = model.distance(clue, mystery_word, language, model_path)
        if dist is not None:
            originality_scores.append(dist)

    originality_str = repr(originality_scores)

    # Statistiken für ALLE eigenen Clues
    if len(originality_scores) > 0:
        min_orig = int(np.min(originality_scores))
        max_orig = int(np.max(originality_scores))
        avg_orig = float(np.mean(originality_scores))
        std_orig = float(np.std(originality_scores, ddof=0))
    else:
        min_orig = max_orig = avg_orig = std_orig = np.nan

    # Originalität NUR für CluesNotInOtherFile
    originality_not_in_other = []
    for clue in clues_not_in_other:
        dist = model.distance(clue, mystery_word, language, model_path)
        if dist is not None:
            originality_not_in_other.append(dist)

    if len(originality_not_in_other) > 0:
        min_orig_not = int(np.min(originality_not_in_other))
        max_orig_not = int(np.max(originality_not_in_other))
        avg_orig_not = float(np.mean(originality_not_in_other))
        std_orig_not = float(np.std(originality_not_in_other, ddof=0))
    else:
        min_orig_not = max_orig_not = avg_orig_not = std_orig_not = np.nan

    # Reihenfolge der Spalten:
    # 1) Mystery Word
    # 2) AllClues
    # 3) ClueFrequency
    # 4) UniqueClues
    # 5) OriginalityScores (alle)
    # 6–9) Stats (alle)
    # 10) CluesNotInOtherFile
    # 11–14) Stats nur für CluesNotInOtherFile
    return pd.Series({
        'Mystery Word': mystery_raw,
        'AllClues': clues_str,
        'ClueFrequency': freq_str,
        'UniqueClues': unique_clues_str,
        'OriginalityScores': originality_str,
        'MinOriginality': min_orig,
        'MaxOriginality': max_orig,
        'AverageOriginality': avg_orig,
        'StdOriginality': std_orig,
        'CluesNotInOtherFile': clues_not_in_other_str,
        'MinOriginalityNotInOther': min_orig_not,
        'MaxOriginalityNotInOther': max_orig_not,
        'AverageOriginalityNotInOther': avg_orig_not,
        'StdOriginalityNotInOther': std_orig_not
    })


# --------------------------------------------------
# Hauptfunktion
# --------------------------------------------------

def main():
    print("Lese Haupt-CSV...")
    df = pd.read_csv(EXCEL_PATH)

    missing = [c for c in ['Mystery Word'] + CLUE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in der Haupt-CSV: {missing}")

    print("Lese andere CSV...")
    other_df = pd.read_csv(OTHER_CSV_PATH)

    if OTHER_CLUE_COLUMN not in other_df.columns:
        raise ValueError(f"In der anderen CSV wurde die Spalte '{OTHER_CLUE_COLUMN}' nicht gefunden.")

    print("Baue Mapping für Clues aus anderer Datei...")
    other_clues_map = build_other_clues_map(other_df, OTHER_CLUE_COLUMN)

    model = Model()

    print("Berechne Originalität & Clue-Vergleiche (mit Vector-Cache)...")
    result_df = df.apply(
        lambda row: process_row(row, model, CLUE_COLUMNS, LANGUAGE, FASTTEXT_MODEL_PATH, other_clues_map),
        axis=1
    )

    print("Schreibe Output...")
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Fertig! Datei gespeichert unter:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
