# -*- coding: utf-8 -*-
"""

@author: Agata 
"""

#Before running, adjust 

#INPUT_FILE   = Path(r"C:\Users\[...]\just_one_clues_random_human_sample.csv")
#OUTPUT_FILE  = Path(r"C:\Users\[...]\just_one_chatgpt_guesses_unique_clues_humans.csv")

#adjust  Game / model configuration if necessary for correct Column names for clues 
#CLUE_COL_NAME   = "UniqueClues" oder "ClueFrequency"
# depending on the elimination rule of clues. if you want to eliminate 
# duplicats completele or just leave one of the duplicats.
# ============================================

import os
import json
import ast
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ============================================
# API key configuration
# ============================================

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable.")

client = OpenAI(api_key=API_KEY)

# ============================================
# File paths
# ============================================

#INPUT_FILE   = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\saved\just_one_clues_hightemp_allmodels_originality.csv")
#OUTPUT_FILE  = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses.csv")


#INPUT_FILE   = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\saved\just_one_clues_hightemp_allmodels_originality.csv")
#OUTPUT_FILE  = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses_unique_clues.csv")

INPUT_FILE   = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_clues_random_human_sample.csv")
OUTPUT_FILE  = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_chatgpt_guesses_unique_clues_humans.csv")


SUMMARY_JSONL = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_summary.jsonl")

# ============================================
# Game / model configuration
# ============================================

MODEL_NAME      = "gpt-4o"    # adjust if needed
FIRST_COL_NAME  = "Mystery Word"   # name of the mystery word column if present
CLUE_COL_NAME   = "UniqueClues"  # column with dict-like clue string

MEMORY_MODE     = "stateless"           # "full", "rolling", "stateless"
ROLLING_WINDOW  = 8
TEMPERATURE     = 1.0
MAX_ATTEMPTS    = 3                # max guess attempts per mystery word

CSV_FLUSH_EVERY_N = 20             # how many new rows before flush to CSV

# Limit how many input rows to process (None = all)
TEST_LIMIT      = None            # e.g. None or 20 for quick tests

OUTPUT_FIELDNAMES = ["mystery_word", "clues", "guesses", "attempts_used"]

# ============================================
# Runtime / safety parameters (per REQUEST)
# ============================================

# Delay after each successful API request (seconds)
REQUEST_DELAY_SEC = 0.4

# Retry logic for failed API calls (network / transient errors)
MAX_RETRIES  = 5      # number of times to retry a failed request
BACKOFF_BASE = 1.8    # exponential backoff base, e.g. 1.8^attempt

# Token monitoring PER REQUEST
# (model context limit is per request, not per run)
REQUEST_TOKEN_WARNING = 80_000    # warn when a single request uses this many tokens
REQUEST_TOKEN_LIMIT   = 128_000   # hard limit per request (close to model context limit)
# How to handle rows where the clue cell is empty or cannot be parsed
# Options:
#   "skip"    → skip this mystery word entirely (no row in output)
#   "invalid" → write a row with attempts_used="invalid_no_clues" and no guesses
#   "noclues" → still call the model, but tell it "No clues available."
EMPTY_CLUE_BEHAVIOR = "invalid"   # or "skip", or "noclues"


# ============================================
# System prompt
# ============================================

SYSTEM_PROMPT = (
    "Just One is a cooperative party game in which you play together to discover as many mystery words as possible."
    "Find the best clue to help your teammate. Be unique, as all identical clues will be canceled!"
    "You have the choice – make the difference!"    
    "Gameplay: There is one active player, who must guess the Mystery Word, and the remaining players, who are the clue givers."
    "A random word will be selected and shown to the clue givers. The corresponding word is this turn's Mystery Word that the active player will have to guess."    
    "Each clue giver secretly writes a clue. The active player can then guess the Mystery Word from the clues."
    "To do so, they can only make three guesses. If the first guess is correct, the team gets 3 points."
    "A correct answer on the second guess yields 2 points. A correct guess on the last try gives 1 point."
    "Three incorrect guesses result in 0 points."   
    "Count the number of points. The goal is to get a high score! The more words your team wins, the better!"   
    "These were the instructions of the game. You will play this game in the role of the guesser."
    "The game plays over several turns. The individual Mystery Words have no connection to one another in any capacity"
    "Treat each bundle of clues as a detached entity."
)

# ============================================
# Helper functions
# ============================================

def build_messages(memory_mode, base_messages, new_user_content, rolling_window):
    """
    Build the message list for the API call.
    base_messages: previous dialogue messages (without system prompt)
    new_user_content: user text for the current attempt
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory_mode == "stateless":
        messages.append({"role": "user", "content": new_user_content})
        return messages

    history = base_messages.copy()
    history.append({"role": "user", "content": new_user_content})

    if memory_mode == "full":
        messages.extend(history)
        return messages

    if memory_mode == "rolling":
        if len(history) > rolling_window:
            history = history[-rolling_window:]
        messages.extend(history)
        return messages

    # Fallback: stateless
    messages.append({"role": "user", "content": new_user_content})
    return messages


def call_model_once(messages, model_name=MODEL_NAME, temperature=TEMPERATURE):
    """
    Single API call without retry.
    Returns (guess_text, usage).
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    guess_text = response.choices[0].message.content.strip()
    usage = response.usage
    return guess_text, usage


def call_model_with_retry(messages, model_name=MODEL_NAME, temperature=TEMPERATURE):
    """
    Call the chat model with retry, delay and per-request token checks.
    Uses:
      - MAX_RETRIES
      - BACKOFF_BASE
      - REQUEST_DELAY_SEC
      - REQUEST_TOKEN_WARNING
      - REQUEST_TOKEN_LIMIT
    Returns (guess_text, usage).
    Raises the last exception if all retries fail.
    """
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            guess_text, usage = call_model_once(
                messages,
                model_name=model_name,
                temperature=temperature
            )

            # Per-request token usage check
            # total_tokens = prompt_tokens + completion_tokens
            req_tokens = usage.total_tokens

            if req_tokens >= REQUEST_TOKEN_LIMIT:
                raise RuntimeError(
                    f"Token limit per request exceeded: {req_tokens} >= {REQUEST_TOKEN_LIMIT}"
                )

            if req_tokens >= REQUEST_TOKEN_WARNING:
                print(f"[WARN] High token usage in this request: {req_tokens} tokens")

            # Delay after successful request (niceness / rate control)
            if REQUEST_DELAY_SEC and REQUEST_DELAY_SEC > 0:
                time.sleep(REQUEST_DELAY_SEC)

            return guess_text, usage

        except Exception as e:
            last_exception = e
            print(f"[WARN] API call failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                print("[ERROR] Max retries reached. Giving up on this call.")
                raise
            # Exponential backoff before next attempt
            sleep_time = BACKOFF_BASE ** (attempt - 1)
            print(f"[INFO] Sleeping {sleep_time:.2f} seconds before retry...")
            time.sleep(sleep_time)

    # Safety fallback
    raise last_exception if last_exception else RuntimeError("Unknown error in call_model_with_retry")


def parse_clue_string(clue_str):
    """
    Parse a clue string into a list of strings.
    Accepts formats like:
      - {'dialect': 2, 'pronunciation': 1}
      - ['pronunciation', 'inflection']
    Returns a list of clue words (strings), or [] if parsing fails.
    """
    if pd.isna(clue_str) or not isinstance(clue_str, str) or not clue_str.strip():
        return []

    try:
        parsed = ast.literal_eval(clue_str)

        # Case 1: dictionary → use keys
        if isinstance(parsed, dict):
            return list(parsed.keys())

        # Case 2: list → use items as strings
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

        # Other types → ignore
        return []

    except (SyntaxError, ValueError):
        return []



def load_input_dataframe(input_path: Path, test_limit=None) -> pd.DataFrame:
    """
    Load the input CSV into a pandas DataFrame and prepare:
    - Ensure there is a column 'mystery_word'
    - Parse ClueFrequency column into 'clue_words' (list of strings)
    - Optionally limit to the first 'test_limit' rows
    """
    df = pd.read_csv(input_path)

    # Ensure we have a "mystery_word" column name
    if FIRST_COL_NAME in df.columns:
        if FIRST_COL_NAME != "mystery_word":
            df = df.rename(columns={FIRST_COL_NAME: "mystery_word"})
    else:
        # Fallback: treat first column as mystery_word
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "mystery_word"})

    # Ensure ClueFrequency column exists (if not, create empty)
    if CLUE_COL_NAME not in df.columns:
        df[CLUE_COL_NAME] = ""

    # Parse clue strings to list of clue words
    df["clue_words"] = df[CLUE_COL_NAME].apply(parse_clue_string)

    # Optional test limit
    if test_limit is not None:
        df = df.head(test_limit)

    return df


def init_jsonl_log(jsonl_path: Path):
    """
    Initialize (empty) the JSONL log file for this run.
    This overwrites any existing file with the same name.
    """
    with jsonl_path.open("w", encoding="utf-8") as f:
        pass  # create or truncate


def append_jsonl_record(jsonl_path: Path, record: dict):
    """
    Append a single JSON record as one line to the JSONL file.
    """
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_output_header(output_path: Path):
    """
    Ensure the output CSV exists with the correct header.
    If the file does not exist, create it with header only.
    If it exists, leave it as is.
    """
    if not output_path.exists():
        header_df = pd.DataFrame(columns=OUTPUT_FIELDNAMES)
        header_df.to_csv(output_path, index=False)


def flush_output_buffer(output_path: Path, buffer):
    """
    Flush the buffered rows (list of dicts) to the output CSV.
    Appends to the file, without header (header already ensured).
    """
    if not buffer:
        return
    df_buffer = pd.DataFrame(buffer)
    df_buffer.to_csv(output_path, mode="a", header=False, index=False)
    buffer.clear()

# ============================================
# Main logic (game with resume based on CSV)
# ============================================

def play_game():
    # Load input data
    df = load_input_dataframe(INPUT_FILE, test_limit=TEST_LIMIT)

    # Determine how many rows are already done based on existing output CSV
    if OUTPUT_FILE.exists():
        done_df = pd.read_csv(OUTPUT_FILE)
        start_index = len(done_df)  # number of completed mystery words
        print(f"Found existing output with {start_index} rows. Resuming from there.")
    else:
        start_index = 0
        print("No existing output CSV found. Starting from row 0.")

    # If all rows already processed, we can stop early
    if start_index >= len(df):
        print("All rows in the input are already processed. Nothing to do.")
        return

    # We're going to process df from start_index onwards
    df_to_process = df.iloc[start_index:].copy()

    # Ensure output CSV header exists
    ensure_output_header(OUTPUT_FILE)

    # Initialize JSONL log for this run (overwrites previous run's JSONL)
    init_jsonl_log(SUMMARY_JSONL)

    # Token summary for this run (for cost analysis; not for limits)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_guesses = 0

    # Counters
    rounds_processed_this_run = 0

    # Buffer for batched CSV writing (new rows only)
    csv_buffer = []

    try:
        for local_row_index, (idx, row) in enumerate(df_to_process.iterrows(), start=start_index):
            mystery_word = str(row["mystery_word"])
            clue_words = row["clue_words"] if isinstance(row["clue_words"], list) else []

            # Handle empty clue list based on EMPTY_CLUE_BEHAVIOR
            if not clue_words:
                if EMPTY_CLUE_BEHAVIOR == "skip":
                    # Completely skip this mystery word
                    print(f"Skipping '{mystery_word}' because it has no clues.")
                    continue

                elif EMPTY_CLUE_BEHAVIOR == "invalid":
                    # Mark this word as invalid and do not call the model
                    out_row = {
                        "mystery_word": mystery_word,
                        "clues": "{}",
                        "guesses": "{}",
                        "attempts_used": "invalid_no_clues"
                    }
                    csv_buffer.append(out_row)
                    rounds_processed_this_run += 1
                    if len(csv_buffer) >= CSV_FLUSH_EVERY_N:
                        flush_output_buffer(OUTPUT_FILE, csv_buffer)
                    continue

                elif EMPTY_CLUE_BEHAVIOR == "noclues":
                    # We still play, but clearly tell the model there are no clues
                    clues_listed = "No clues available."
                else:
                    # Unknown setting → default to "noclues"
                    clues_listed = "No clues available."
            else:
                # Normal case: we have clue words
                clues_listed = "\n".join(
                    f"{i + 1}) {w}" for i, w in enumerate(clue_words)
                )

            base_messages = []

            base_clues_text = (
                "Here are clue words for a secret mystery word:\n"
                f"{clues_listed}\n\n"
                "Answer with ONLY one word: your guess for the mystery word."
            )

            guesses = []
            attempts_used = None

            for attempt in range(1, MAX_ATTEMPTS + 1):
                if attempt == 1:
                    # First attempt: only show clues and main instruction
                    user_text = base_clues_text
                else:
                    # For second and third attempts: explicitly show previous incorrect guesses
                    previous_guesses_str = (
                        ", ".join(f"'{g}'" for g in guesses) if guesses else "none"
                    )

                    user_text = (
                        base_clues_text
                        + f"\n\nYour previous incorrect guesses: {previous_guesses_str}."
                        + "\nThe last guess was wrong. Try again."
                    )

                messages = build_messages(
                    MEMORY_MODE,
                    base_messages,
                    user_text,
                    ROLLING_WINDOW
                )

                # Call model with retry, per-request token checks and delay
                guess, usage = call_model_with_retry(
                    messages,
                    model_name=MODEL_NAME,
                    temperature=TEMPERATURE
                )
                guesses.append(guess)

                # Accumulate token usage (for run-level stats only)
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                round_total_tokens = usage.total_tokens

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += round_total_tokens
                total_guesses += 1

                # Extend base_messages for memory modes (without system)
                base_messages.append({"role": "user", "content": user_text})
                base_messages.append({"role": "assistant", "content": guess})

                # JSONL logging for this call
                call_record = {
                    "type": "call",
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "round_index": local_row_index,
                    "mystery_word": mystery_word,
                    "attempt": attempt,
                    "memory_mode": MEMORY_MODE,
                    "model": MODEL_NAME,
                    "input_text": user_text,
                    "output_text": guess,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": round_total_tokens,
                }
                append_jsonl_record(SUMMARY_JSONL, call_record)

                # Check if guess is correct
                if guess.lower().strip() == mystery_word.lower().strip():
                    attempts_used = attempt
                    break

            # Build clue string as: {'dialect', 'pronunciation', 'inflection'}
            if clue_words:
                clues_str = "{" + ", ".join(f"'{w}'" for w in clue_words) + "}"
            else:
                clues_str = "{}"

            # Build guesses string as: {'guess1', 'guess2', ...}
            if guesses:
                guesses_str = "{" + ", ".join(f"'{g}'" for g in guesses) + "}"
            else:
                guesses_str = "{}"

            out_row = {
                "mystery_word": mystery_word,
                "clues": clues_str,
                "guesses": guesses_str,
                "attempts_used": attempts_used if attempts_used is not None else "None"
            }

            csv_buffer.append(out_row)
            rounds_processed_this_run += 1

            # Flush buffer every N rows
            if len(csv_buffer) >= CSV_FLUSH_EVERY_N:
                flush_output_buffer(OUTPUT_FILE, csv_buffer)

    except KeyboardInterrupt:
        print("\nInterrupted by user (KeyboardInterrupt).")
    finally:
        # Flush remaining rows in buffer
        flush_output_buffer(OUTPUT_FILE, csv_buffer)

        # Summary record for this run (not for all runs ever, only this execution)
        summary_record = {
            "type": "summary",
            "timestamp_utc": datetime.utcnow().isoformat(),
            "model": MODEL_NAME,
            "memory_mode": MEMORY_MODE,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "rounds_total_planned": len(df),
            "rounds_already_done_before_run": start_index,
            "rounds_processed_this_run": rounds_processed_this_run,
            "rounds_total_done_after_run": start_index + rounds_processed_this_run,
            "guesses_total_this_run": total_guesses,
            "request_token_warning": REQUEST_TOKEN_WARNING,
            "request_token_limit": REQUEST_TOKEN_LIMIT,
            "output_csv": str(OUTPUT_FILE),
        }
        append_jsonl_record(SUMMARY_JSONL, summary_record)


if __name__ == "__main__":
    play_game()
    print("Game finished.")
    print("Output CSV :", OUTPUT_FILE)
    print("JSONL log  :", SUMMARY_JSONL)
