# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:36:01 2025

@author: serge
"""

# -*- coding: utf-8 -*-
# Skript für lange Datei: eine Zeile = (mystery_word, clue)

import re
import math
import unicodedata
import numpy as np
import pandas as pd
import fasttext
from tqdm import tqdm
import scipy.spatial.distance as sdist

# ================== PFAD-KONFIGURATION ==================
INPUT_CSV  = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\data_merged_clues_ai.csv"
OUTPUT_CSV = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\data_merged_clues_ai_with_originality.csv"

FASTTEXT_MODEL_PATH = r"C:\Users\serge\Documents\Thesis\Data Seafile\Code\cc.en.300.bin"
# ========================================================
# -*- coding: utf-8 -*-
# Skript für lange Datei: eine Zeile = (mystery_word, clue)

# ========================================================

# -------- Text-Cleaning & Checks --------

_punct_regex = re.compile(r"[^\w\s\-]")  # alles außer Wortzeichen, Leerraum, Bindestrich

def clean_token(s: str) -> str:
    """Lowercase + Satzzeichen entfernen + Whitespace normalisieren."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _punct_regex.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def safe_is_invalid(x) -> bool:
    """True, wenn der Wert leer, NaN oder 'ERROR' ist."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str):
        if not x.strip():
            return True
        if x.strip().upper() == "ERROR":
            return True
    return False

def is_gibberish_token(s: str) -> bool:
    """Erkenne kaputte / nicht-lateinische Tokens."""
    if not isinstance(s, str) or not s.strip():
        return True

    s_norm = unicodedata.normalize("NFKD", s)
    s_clean = _punct_regex.sub("", s_norm).strip().lower()

    if not s_clean:
        return True

    # Nur a-z und Bindestrich erlaubt
    if re.search(r"[^a-z\-]", s_clean):
        return True

    return False

# -------- FastText mit Cache --------

class FTModelCache:
    def __init__(self, model_path: str):
        print("Lade FastText-Modell ...")
        self.model = fasttext.load_model(model_path)
        print("FastText-Modell geladen.")
        self.vec_cache = {}

    def get_vec(self, token: str):
        if not token:
            return None
        v = self.vec_cache.get(token)
        if v is not None:
            return v
        try:
            vec = self.model.get_word_vector(token)
            if vec is None:
                return None
            vec = np.asarray(vec, dtype=np.float32)
            self.vec_cache[token] = vec
            return vec
        except Exception:
            return None


# -------- Hauptfunktion --------

def compute_originality_longfile(df: pd.DataFrame,
                                 model_path: str) -> pd.DataFrame:
    required_cols = ["mystery_word", "clues"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in der Eingabedatei: {missing}")

    ft = FTModelCache(model_path)
    mw_vecs = {}

    originality_float = []
    originality_rounded = []
    cleaned_clues_output = []   # <<<<<<<<<<<<<< NEUE SPALTE

    print("Berechne Originalität für lange Datei ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mw_raw = row["mystery_word"]
        clue_raw = row["clues"]

        # Invalid → direkt None
        if safe_is_invalid(mw_raw) or safe_is_invalid(clue_raw):
            cleaned_clues_output.append(None)
            originality_float.append(None)
            originality_rounded.append(None)
            continue

        mw = clean_token(str(mw_raw))
        clue = clean_token(str(clue_raw))

        # Gibberish → None
        if is_gibberish_token(clue):
            cleaned_clues_output.append(None)
            originality_float.append(None)
            originality_rounded.append(None)
            continue

        if not mw or not clue:
            cleaned_clues_output.append(None)
            originality_float.append(None)
            originality_rounded.append(None)
            continue

        # Diesen sauberen Clue speichern
        cleaned_clues_output.append(clue)

        # Mystery-Word-Vektor
        v1 = mw_vecs.get(mw)
        if v1 is None:
            v1 = ft.get_vec(mw)
            mw_vecs[mw] = v1

        v2 = ft.get_vec(clue)

        if v1 is None or v2 is None:
            originality_float.append(None)
            originality_rounded.append(None)
            continue

        try:
            d_raw = sdist.cosine(v1, v2) * 100.0
            if math.isnan(d_raw):
                originality_float.append(None)
                originality_rounded.append(None)
                continue

            originality_float.append(d_raw)
            originality_rounded.append(int(round(d_raw)))

        except Exception:
            originality_float.append(None)
            originality_rounded.append(None)
            continue

    df["cleaned_clue"] = cleaned_clues_output        # <<<<<< NEUE SPALTE
    df["originality_index"] = originality_float
    df["originality_index_rounded"] = originality_rounded

    return df


if __name__ == "__main__":
    print("Lese Eingabe-CSV ...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8", sep=",", engine="python")

    df_out = compute_originality_longfile(df, FASTTEXT_MODEL_PATH)

    print(f"Speichere Ergebnis nach: {OUTPUT_CSV}")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Fertig.")
