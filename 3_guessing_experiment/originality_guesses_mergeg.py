# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:06:29 2025

@author: serge
"""
import re
import math
import unicodedata
import numpy as np
import pandas as pd
import fasttext
from tqdm import tqdm
import scipy.spatial.distance as sdist
import ast  

# ================== PFAD-KONFIGURATION ==================
INPUT_CSV  = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\guesses_combined_cleaned.csv"
OUTPUT_CSV = r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\guesses_combined_cleaned_with_originality.csv"

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

    # NEUE Listen für Output
    clues_list_output = []                  # z.B. ['berry', 'fruit', ...]
    originality_list_float_output = []      # z.B. [4.123, 5.89, ...]
    originality_list_rounded_output = []    # z.B. [4, 6, 6]

    print("Berechne Originalität für lange Datei ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mw_raw = row["mystery_word"]
        clues_raw = row["clues"]

        # Invalid → direkt None
        if safe_is_invalid(mw_raw) or safe_is_invalid(clues_raw):
            clues_list_output.append(None)
            originality_list_float_output.append(None)
            originality_list_rounded_output.append(None)
            continue

        mw = clean_token(str(mw_raw))
        if not mw:
            clues_list_output.append(None)
            originality_list_float_output.append(None)
            originality_list_rounded_output.append(None)
            continue

        # Mystery-Word-Vektor
        v1 = mw_vecs.get(mw)
        if v1 is None:
            v1 = ft.get_vec(mw)
            mw_vecs[mw] = v1

        # -------- clues-Spalte in eine Liste umwandeln --------
        # Fälle:
        #  - String wie "{'a', 'b', 'c'}"
        #  - echtes set/list/tuple
        #  - einfacher String ohne Klammern
        clues_iterable = None

        if isinstance(clues_raw, str):
            s = clues_raw.strip()
            # Versuch, Set-String zu parsen
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (set, list, tuple)):
                    clues_iterable = list(parsed)
                else:
                    # war z.B. nur ein einfacher String
                    clues_iterable = [clues_raw]
            except (ValueError, SyntaxError):
                # kein gültiges Python-Literal → als einzelner Clue behandeln
                clues_iterable = [clues_raw]
        elif isinstance(clues_raw, (set, list, tuple)):
            clues_iterable = list(clues_raw)
        else:
            clues_iterable = [clues_raw]

        row_clues_clean = []
        row_orig_float = []
        row_orig_round = []

        # Für jeden einzelnen Clue in der Zeile
        for clue in clues_iterable:
            clue = clean_token(str(clue))

            if not clue or is_gibberish_token(clue):
                continue  # kaputte/leer Clues überspringen

            row_clues_clean.append(clue)

            v2 = ft.get_vec(clue)
            if v1 is None or v2 is None:
                row_orig_float.append(None)
                row_orig_round.append(None)
                continue

            try:
                d_raw = sdist.cosine(v1, v2) * 100.0
                if math.isnan(d_raw):
                    row_orig_float.append(None)
                    row_orig_round.append(None)
                else:
                    row_orig_float.append(d_raw)
                    row_orig_round.append(int(round(d_raw)))
            except Exception:
                row_orig_float.append(None)
                row_orig_round.append(None)

        # Wenn nach dem Filtern keine Clues übrig sind
        if not row_clues_clean:
            clues_list_output.append(None)
            originality_list_float_output.append(None)
            originality_list_rounded_output.append(None)
        else:
            clues_list_output.append(row_clues_clean)
            originality_list_float_output.append(row_orig_float)
            originality_list_rounded_output.append(row_orig_round)

    # NEUE Spalten
    df["clues_list"] = clues_list_output
    df["originality_index_list"] = originality_list_float_output
    df["originality_index_rounded_list"] = originality_list_rounded_output

    return df



if __name__ == "__main__":
    print("Lese Eingabe-CSV ...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8", sep=",", engine="python")

    df_out = compute_originality_longfile(df, FASTTEXT_MODEL_PATH)

    print(f"Speichere Ergebnis nach: {OUTPUT_CSV}")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Fertig.")
