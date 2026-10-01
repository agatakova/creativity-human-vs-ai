# -*- coding: utf-8 -*-
"""

@author: Agata
"""

#Before running, adjust 

#MEMORY_MODE   = "STATELESS",
#INPUT_FILE   = Path(r"C:\Users\[...]\bga_data_en.csv")
#OUTPUT_FILE  = Path(r"C:\Users\[...]\Output\just_one_clues_gemini.csv")
#MODEL = "gemini-2.5-pro" or "gemini-2.5-flash"

#It is not recomendet by the author to include all temperatures into one run, 
#due to possible crashes.
import os
import time
import json
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections.abc import Mapping

# =========================================================================
# (1) Import Google Gen AI SDK
# =========================================================================
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError as GenAI_APIError
except ImportError:
    raise ImportError(
        "The Google Gen AI SDK is not installed. Run 'pip install google-genai'."
    )

# =========================================================================
# (2) API Key configuration
# =========================================================================
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY as an environment variable.")

# =========================================================================
# (3) Model configuration
# =========================================================================
MODEL = "gemini-2.5-pro"
PRICE_INPUT_PER_MTOK = 1.25
PRICE_OUTPUT_PER_MTOK = 10.0

MEMORY_MODE = "FULL"
MEMORY_WINDOW = 10

SAVE_RAW_JSON = True
TEMPERATURES: Dict[str, float] = {
    #"ClueTemperature_0_0": 0.0,
    #"ClueTemperature_0_25": 0.25,
    #"ClueTemperature_0_5": 0.5,
    #"ClueTemperature_0_75": 0.75,
    #"ClueTemperature_0_7": 0.7,
    #"ClueTemperature_1_0": 1.0,
    #"ClueTemperature_1_2": 1.2,
    #"ClueTemperature_1_25": 1.25,
    "ClueTemperature_1_5": 1.5,
    #"ClueTemperature_1_75": 1.75,
    #"ClueTemperature_2_0": 2.0,
    
}
INPUT_FILE = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv")
OUTPUT_FILE = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_clues_gemini_temp_1_2_pro.csv")
RAW_JSONL_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_raw.jsonl")

MAX_ROWS = None
CHECKPOINT_EVERY = 20
REQUEST_DELAY_SEC = 0.4
MAX_RETRIES = 4
BACKOFF_BASE = 1.8
TOKEN_WARNING = 100_000
TOKEN_LIMIT = 1_000_000

SYSTEM_PROMPT = (
    "Just One is a cooperative party game in which you play together to discover as many mystery words as possible. "
    "Find the best clue to help your teammate. Be unique, as all identical clues will be cancelled! You have the choice – make the difference! "
    "Gameplay: There is one active player, who must guess the Mystery Word, and the remaining players, who are the clue givers. "
    "A random word will be selected and shown to the clue givers. This word is the Mystery Word for the current turn, which the active player will have to guess. "
    "Each clue giver secretly writes a clue. This clue must consist of a single word written only with letters (A–Z) — no numbers, no symbols, special characters, or punctuation. "
    "Compound expressions or multiple words are not valid. Only real, correctly spelled words that exist in the English word list are accepted. "
    "Invalid clues are, for example, the Mystery Word itself or the same word written in a different language (Vert to guess Green), "
    "a word in the same family as the Mystery Word (Princess to guess Prince, Japanese to guess Japan), "
    "a made-up word (Cuppajo to guess Coffee), a homophone of the Mystery Word (Whether to guess Weather), "
    "the same words in the singular and plural (Horse and Horses) or the same words with spelling mistakes (Rhythm and Rythem). "
    "Any invalid or identical clues are canceled. "
    "After all invalid or identical clues have been removed, the active player can now guess the Mystery Word from the remaining clues. "
    "A correct guess results in won word, otherwise a lost word. "
    "The goal is to get a high score of won word cards compared to lost ones! The more words your team won, the better! "
    "These were the instructions of the game. You will play this game as one of the clue givers, in a team with other AI chat bots powered by large language models. "
    "The game plays over several turns, and each Mystery Word is completely independent of the others. "
    "Treat each Mystery Word as a detached entity"
)

USER_PROMPT_FIRST = 'Here is your first Mystery word: "{mystery}". Respond with one valid one-word clue.'
USER_PROMPT_NEXT = 'The next Mystery word is: "{mystery}". Respond with one valid one-word clue.'

# =========================================================================
# (4) Create Gemini client
# =========================================================================
client = genai.Client(api_key=API_KEY)

# =========================================================================
# Helper functions
# =========================================================================

def backoff_sleep(attempt: int):
    """Wait longer after each failed attempt."""
    time.sleep((BACKOFF_BASE ** attempt) + (0.05 * attempt))
    
import re

# Accepts letters incl. accents; optionally allows hyphen or apostrophe inside the word
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?")

def extract_clue_word(text: str) -> str:
    """
    Return a single clue word from model output.
    - If 'THOUGHTS' is present (any case), return the LAST plausible word.
    - Otherwise, return the FIRST plausible word.
    - If none found, return 'INVALID'.
    """
    if not text:
        return "INVALID"

    words = WORD_RE.findall(text)
    if not words:
        return "INVALID"

    # If model spilled chain-of-thought, take the last word (the actual chosen clue is usually at the end)
    if "THOUGHTS" in text.upper():
        return words[-1]

    # Otherwise, keep it strict: only the first word
    return words[0]


def trim_messages_for_mode(messages: List[Dict], mode: str, window_size: int = 10) -> List[Dict]:
    """Trim chat history depending on the configured memory mode."""
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    others = [m for m in messages if m.get("role") != "system"]

    mu = mode.upper()
    if mu == "FULL":
        return messages
    base = [sys_msg] if sys_msg else []
    if mu == "STATELESS":
        last_user = next((m for m in reversed(others) if m.get("role") == "user"), None)
        return base + ([last_user] if last_user else [])
    if mu == "ROLLING":
        return base + others[-window_size:]
    return messages


# =========================================================================
# (5) Chat function using Gemini generate_content
# =========================================================================
def chat_once(messages: List[Dict], temperature: float):
    """Send one chat request using Gemini."""
    gemini_messages: List[types.Content] = []
    system_instruction: str = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not content:
            continue
        if role == "system":
            system_instruction = content
            continue

        gemini_role = "model" if role == "assistant" else role
        gemini_messages.append(
            types.Content(role=gemini_role, parts=[types.Part(text=content)])
        )

    # Do not include max_output_tokens; it may suppress responses in current SDK versions
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction or None,
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=gemini_messages,
        config=generation_config,
    )

    text = (resp.text or "").strip()

    usage_metadata = getattr(resp, "usage_metadata", None)
    if usage_metadata:
        usage_dict = {
            "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
        }
    else:
        usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    return text, usage_dict, resp


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD based on token usage."""
    return (tokens_in / 1_000_000) * PRICE_INPUT_PER_MTOK + (tokens_out / 1_000_000) * PRICE_OUTPUT_PER_MTOK


def save(out_df: pd.DataFrame):
    """Save DataFrame to CSV without sorting."""
    out_df.to_csv(OUTPUT_FILE, encoding="utf-8", index=True)


def log_raw_json(word: str, label: str, full_response: Any, usage_dict: Dict[str, int]):
    """Save raw API responses to JSONL."""
    if not SAVE_RAW_JSON or full_response is None:
        return
    if not RAW_JSONL_FILE.exists():
        RAW_JSONL_FILE.touch()

    if hasattr(full_response, "to_dict"):
        response_payload = full_response.to_dict()
    elif isinstance(full_response, Mapping):
        response_payload = dict(full_response)
    else:
        response_payload = str(full_response)

    log_entry = {
        "word": word,
        "temperature": label,
        "response": response_payload,
        "usage": usage_dict,
    }
    with RAW_JSONL_FILE.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")


def save_run_summary_json(
    total_in_tokens: int,
    total_out_tokens: int,
    *,
    started_at_iso: str,
    ended_at_iso: str,
):
    """Write a summary JSON file for the run."""
    summary_file = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_run_summary.json")

    input_usd = (total_in_tokens / 1_000_000) * PRICE_INPUT_PER_MTOK
    output_usd = (total_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK
    total_usd = input_usd + output_usd

    summary_entry = {
        "type": "run_summary",
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "temperatures": TEMPERATURES,
        "input_tokens": total_in_tokens,
        "output_tokens": total_out_tokens,
        "total_tokens": total_in_tokens + total_out_tokens,
        "cost": {
            "input_usd": round(input_usd, 6),
            "output_usd": round(output_usd, 6),
            "total_usd": round(total_usd, 6),
        },
        "files": {
            "input_csv": str(INPUT_FILE),
            "output_csv": str(OUTPUT_FILE),
            "raw_jsonl": str(RAW_JSONL_FILE),
            "summary_json": str(summary_file),
        },
        "started_at": started_at_iso,
        "ended_at": ended_at_iso,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_entry, f, indent=4, ensure_ascii=False)

    print(f"Run summary saved → {summary_file}")


def user_prompt(is_first: bool, mystery: str) -> str:
    """Generate user prompt for the given mystery word."""
    return (
        USER_PROMPT_FIRST.format(mystery=mystery)
        if is_first
        else USER_PROMPT_NEXT.format(mystery=mystery)
    )


def request_raw_with_memory(messages: List[Dict], mystery: str, temperature: float):
    """
    Send one prompt with retries and memory trimming.
    Appends only the cleaned one-word clue (not the model's full output) to the conversation history.
    Returns: (clue_word, usage_dict, messages, full_response, raw_model_output)
    """
    is_first_turn = (sum(1 for m in messages if m["role"] == "assistant") == 0)
    messages.append({"role": "user", "content": user_prompt(is_first_turn, mystery)})
    call_messages = trim_messages_for_mode(messages, MEMORY_MODE, MEMORY_WINDOW)

    attempt = 0
    last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_dict: Dict[str, int] = last_usage

    while attempt < MAX_RETRIES:
        try:
            # raw_model_output is the verbatim text from the model (may include THOUGHTS etc.)
            raw_model_output, usage_dict, full_response = chat_once(call_messages, temperature)
            last_usage = usage_dict

            # Clean it down to the actual clue word
            clue_word = extract_clue_word(raw_model_output)

            # Store only the cleaned word in the conversation history
            messages.append({"role": "assistant", "content": clue_word})

            # Return both: the cleaned word (used by the pipeline) and the raw model output (for logging/console)
            return clue_word, usage_dict, messages, full_response, raw_model_output

        except GenAI_APIError as e:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] API Error (Attempt {attempt + 1}): {e}")
            attempt += 1
            backoff_sleep(attempt)

        except Exception as e:
            usage_dict = last_usage
            print(f"[{datetime.now().isoformat(timespec='seconds')}] Generic Error (Attempt {attempt + 1}): {e}")
            attempt += 1
            backoff_sleep(attempt)

    messages.append({"role": "assistant", "content": "ERROR"})
    return "ERROR", last_usage, messages, None, ""


# =========================
# Main logic
# =========================

def main():
    print("Loading input file ...")
    started_at_iso = datetime.now().isoformat(timespec="seconds") 
    df = pd.read_csv(INPUT_FILE, encoding="utf-8", nrows=MAX_ROWS)
    words: List[str] = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    if not words:
        raise RuntimeError("No mystery words found in the input file.")

    if OUTPUT_FILE.exists():
        out_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8").set_index("Mystery Word")
    else:
        cols = ["Mystery Word"] + list(TEMPERATURES.keys())
        out_df = pd.DataFrame(columns=cols).set_index("Mystery Word")

    if SAVE_RAW_JSON and not RAW_JSONL_FILE.exists():
        RAW_JSONL_FILE.touch()

    total = len(words)
    print(f"Starting generation for {total} words ...")

    grand_in_tokens = 0
    grand_out_tokens = 0

    try:
        for label, temp in TEMPERATURES.items():
            print(f"\nTemperature: {label} (T={temp}) – Mode: {MEMORY_MODE} (Window={MEMORY_WINDOW})")

            messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            processed_since_checkpoint = 0

            for idx, word in enumerate(words, start=1):
                if word in out_df.index and pd.notna(out_df.loc[word, label]):
                    print(f"[{idx}/{total}] {word} – already present, skipping.")
                    continue

                print(f"[{idx}/{total}] {word} | {label} → sending request ...")
                clue, usage, messages, full_response, raw_model = request_raw_with_memory(messages, word, temp)

                candidate = (clue or "").strip()
                if (not candidate) or (candidate.lower() in {"error", "none", ""}):
                    candidate = "INVALID"
                
                out_df.loc[word, label] = candidate
                
                in_t = usage.get("input_tokens") or 0
                out_t = usage.get("output_tokens") or 0
                grand_in_tokens += in_t
                grand_out_tokens += out_t
                
                tot_t = usage.get("total_tokens", 0)
                if tot_t > TOKEN_WARNING:
                    print(f"Warning: context size {tot_t} tokens (limit ~{TOKEN_LIMIT})")
                
                # Show both: what the model said vs. what we stored
                print(f"  Model output: {raw_model!r}  | Stored clue: {candidate!r}  | Tokens in/out: {in_t}/{out_t}")
                
                log_raw_json(word, label, full_response, usage)


                time.sleep(REQUEST_DELAY_SEC)

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    save(out_df)
                    print(f"Checkpoint saved → {OUTPUT_FILE}")
                    processed_since_checkpoint = 0

            save(out_df)
            print(f"Completed {label}. Results saved to CSV.")

    except KeyboardInterrupt:
        print("\nInterrupted by user – saving current progress ...")
    except Exception as e:
        print(f"\nUnexpected error: {e}\nSaving current progress ...")
    finally:
        save(out_df)

        total_in_usd = (grand_in_tokens / 1_000_000) * PRICE_INPUT_PER_MTOK
        total_out_usd = (grand_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK

        ended_at_iso = datetime.now().isoformat(timespec="seconds")
        save_run_summary_json(
            grand_in_tokens,
            grand_out_tokens,
            started_at_iso=started_at_iso,
            ended_at_iso=ended_at_iso,
        )

        print("\n===== Run Summary =====")
        print(f"Total tokens input/output: {grand_in_tokens}/{grand_out_tokens}")
        print(f"Costs (approx.): ${total_in_usd:.4f} in + ${total_out_usd:.4f} out = ${total_in_usd + total_out_usd:.4f}")
        print("=======================")


if __name__ == "__main__":
    main()
