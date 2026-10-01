# -*- coding: utf-8 -*-
"""

@author: Agata
"""

#Before running, adjust 

#MEMORY_MODE   = "STATELESS",
#INPUT_FILE   = Path(r"C:\Users\[...]\bga_data_en.csv")
#OUTPUT_FILE  = Path(r"C:\Users\[...]\Output\just_one_clues_deepseek.csv")

#It is not recomendet by the author to include all temperatures into one run, 
#due to possible crashes.


import os
import time
import json
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = openai.OpenAI  # type: ignore

# =========================
# Configuration
# =========================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY as an environment variable.")

MODEL = "deepseek-chat"

# Price info (for cost calculation)
PRICE_INPUT_PER_MTOK  = 0.28
PRICE_OUTPUT_PER_MTOK = 0.42

# Conversation memory mode
# "FULL"      = keep full history
# "ROLLING"   = keep only the last N user/assistant messages (system remains)
# "STATELESS" = no history (only system + current user message)
MEMORY_MODE   = "FULL"
MEMORY_WINDOW = 10

# Whether to store full raw API responses in JSONL
SAVE_RAW_JSON = True

# Temperature settings
TEMPERATURES: Dict[str, float] = {
    "ClueTemperature_0_7": 0.7,
    "ClueTemperature_1_2": 1.2,
    "ClueTemperature_1_5": 1.5,
}

# File paths
INPUT_FILE   = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv")
OUTPUT_FILE  = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_clues_deepseek_test2.csv")
RAW_JSONL_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_raw.jsonl")

# Runtime parameters
MAX_ROWS = None
CHECKPOINT_EVERY  = 20
REQUEST_DELAY_SEC = 0.4
MAX_RETRIES       = 4
BACKOFF_BASE      = 1.8
TOKEN_WARNING     = 100_000
TOKEN_LIMIT       = 64_000

# System prompt (paste your final wording here if you have a newer version)
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
    "Treat each Mystery Word as a detached entity."
)

# User message templates
USER_PROMPT_FIRST = 'Here is your first Mystery word: "{mystery}". Respond with one valid one-word clue.'
USER_PROMPT_NEXT  = 'The next Mystery word is: "{mystery}". Respond with one valid one-word clue.'

# Create OpenAI-compatible client pointed at DeepSeek
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# =========================
# Helper functions
# =========================

def backoff_sleep(attempt: int):
    """Wait longer after each failed attempt."""
    time.sleep((BACKOFF_BASE ** attempt) + (0.05 * attempt))

def trim_messages_for_mode(messages: List[Dict], mode: str, window_size: int = 10) -> List[Dict]:
    """
    Keep or reduce chat history depending on memory mode.
    - FULL: keep all messages
    - STATELESS: keep only system + most recent user
    - ROLLING: keep system + last N non-system messages
    """
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

def chat_once(messages: List[Dict], temperature: float):
    """
    Send a single chat request with a hard limit on output length.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=messages,
        max_tokens=15,      # strictly short: one word + minimal buffer
        stop=["\n"],       # stop at the first line break
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    return text, usage, resp

def cost_usd(tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD based on token usage."""
    return (tokens_in / 1_000_000) * PRICE_INPUT_PER_MTOK + (tokens_out / 1_000_000) * PRICE_OUTPUT_PER_MTOK

def save(out_df: pd.DataFrame):
    """Save results to CSV without sorting (preserve input order)."""
    out_df.to_csv(OUTPUT_FILE, encoding="utf-8", index=True)

def log_raw_json(word: str, label: str, full_response, usage_dict: Dict[str, int]):
    """Save the full API response to JSONL if enabled."""
    if not SAVE_RAW_JSON or full_response is None:
        return
    if not RAW_JSONL_FILE.exists():
        RAW_JSONL_FILE.touch()

    response_payload = full_response.to_dict() if hasattr(full_response, "to_dict") else str(full_response)
    log_entry = {
        "word": word,
        "temperature": label,
        "response": response_payload,
        "usage": {
            "input_tokens": usage_dict.get("input_tokens", 0),
            "output_tokens": usage_dict.get("output_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0),
        },
    }
    with RAW_JSONL_FILE.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def save_run_summary_json(
    total_in_tokens: int,
    total_out_tokens: int,
    *,
    started_at_iso: str,
    ended_at_iso: str,
):
    """
    Save a run summary to a standalone JSON file (next to the output CSV).
    Always runs at the end, even after errors.
    """
    summary_file = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_run_summary.json")

    # Calculate cost
    input_usd  = (total_in_tokens  / 1_000_000) * PRICE_INPUT_PER_MTOK
    output_usd = (total_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK
    total_usd  = input_usd + output_usd

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

    # Write summary as a clean, readable JSON file
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_entry, f, indent=4, ensure_ascii=False)

    print(f"Run summary saved → {summary_file}")


def user_prompt(is_first: bool, mystery: str) -> str:
    """Return the correct user prompt based on whether it's the first or a later turn."""
    return (
        USER_PROMPT_FIRST.format(mystery=mystery)
        if is_first
        else USER_PROMPT_NEXT.format(mystery=mystery)
    )

def request_raw_with_memory(messages: List[Dict], mystery: str, temperature: float):
    """
    Send one prompt with retries and memory trimming.
    Appends both the user message and assistant response to the conversation history.
    """
    is_first_turn = (sum(1 for m in messages if m["role"] == "assistant") == 0)
    messages.append({"role": "user", "content": user_prompt(is_first_turn, mystery)})
    call_messages = trim_messages_for_mode(messages, MEMORY_MODE, MEMORY_WINDOW)

    attempt = 0
    last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    while attempt < MAX_RETRIES:
        try:
            raw, usage, full_response = chat_once(call_messages, temperature)
            usage_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
            messages.append({"role": "assistant", "content": raw})
            return raw, usage_dict, messages, full_response
        except Exception:
            attempt += 1
            backoff_sleep(attempt)

    messages.append({"role": "assistant", "content": "ERROR"})
    return "ERROR", last_usage, messages, None

# =========================
# Main logic
# =========================

def main():
    print("Loading input file ...")
    started_at_iso = datetime.now().isoformat(timespec="seconds")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8", nrows=MAX_ROWS) #nrows=MAX_ROWS)
    words: List[str] = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    if not words:
        raise RuntimeError("No mystery words found in the input file.")

    # Load or create output
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
            t_in, t_out = 0, 0
            processed_since_checkpoint = 0

            for idx, word in enumerate(words, start=1):
                # Skip already processed words for this temperature                
                if word in out_df.index and pd.notna(out_df.loc[word, label]):
                    print(f"[{idx}/{total}] {word} – already present, skipping.")
                    continue

                print(f"[{idx}/{total}] {word} | {label} → sending request ...")
                raw, usage, messages, full_response = request_raw_with_memory(messages, word, temp)

                # Basic hard rule for INVALID:
                # - empty/placeholder outputs
                # - OR any newline present in the original raw response
                candidate = (raw or "").strip()
                if (not candidate) or ("\n" in raw) or (candidate.lower() in {"error", "none", ""}):
                    candidate = "INVALID"

                out_df.loc[word, label] = candidate

                in_t = usage.get("input_tokens", 0)
                out_t = usage.get("output_tokens", 0)
                tot_t = usage.get("total_tokens", 0)
                t_in += in_t
                t_out += out_t
                grand_in_tokens += in_t
                grand_out_tokens += out_t

                # Warn if context gets large
                if tot_t > TOKEN_WARNING:
                    print(f"Warning: context size {tot_t} tokens (limit ~{TOKEN_LIMIT})")

                print(f"    Raw answer: {raw!r}  | Final clue saved: {candidate!r}  | Tokens in/out: {in_t}/{out_t}")

                # Write raw JSON log entry
                log_raw_json(word, label, full_response, usage)

                # Pause between requests
                time.sleep(REQUEST_DELAY_SEC)
                
                # Save checkpoint
                processed_since_checkpoint += 1
                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    save(out_df)
                    print(f"Checkpoint saved → {OUTPUT_FILE}")
                    processed_since_checkpoint = 0
                    
            # Save after each temperature
            save(out_df)
            cost_estimate = cost_usd(t_in, t_out)
            print(f"Completed {label}. Tokens in/out: {t_in}/{t_out} | Approx. cost: ${cost_estimate:.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user – saving current progress ...")
    except Exception as e:
        print(f"\nUnexpected error: {e}\nSaving current progress ...")
    finally:
        # Always save the CSV results
        save(out_df)
    
        # Compute total cost
        total_in_usd  = (grand_in_tokens  / 1_000_000) * PRICE_INPUT_PER_MTOK
        total_out_usd = (grand_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK
    
        # Save standalone JSON summary
        ended_at_iso = datetime.now().isoformat(timespec="seconds")
        save_run_summary_json(
            grand_in_tokens,
            grand_out_tokens,
            started_at_iso=started_at_iso,
            ended_at_iso=ended_at_iso,
        )
    
        # Console summary
        print("\n===== Run Summary =====")
        print(f"Total tokens input/output: {grand_in_tokens}/{grand_out_tokens}")
        print(f"Costs (approx.): ${total_in_usd:.4f} in + ${total_out_usd:.4f} out = ${total_in_usd + total_out_usd:.4f}")
        print("=======================")



if __name__ == "__main__":
    main()
