# -*- coding: utf-8 -*-
"""

@author: Agata
"""

#Before running, adjust 

#MEMORY_MODE   = "STATELESS",
#INPUT_FILE   = Path(r"C:\Users\[...]\bga_data_en.csv")
#OUTPUT_FILE  = Path(r"C:\Users\[...]\Output\just_one_clues_claude.csv")

#It is not recomendet by the author to include all temperatures into one run, 
#due to possible crashes.

import os
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd

# === Anthropic SDK ===
try:
    import anthropic
except ImportError:
    raise RuntimeError("Please `pip install anthropic` before running this script.")

# =========================
# Configuration
# =========================

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set ANTHROPIC_API_KEY as an environment variable.")

client = anthropic.Anthropic(api_key=API_KEY)

# Model (you can also use "claude-sonnet-4-20250514")
MODEL = "claude-sonnet-4-5-20250929"

# Price info (update to current Claude pricing if you want accurate totals)
PRICE_INPUT_PER_MTOK = 3.00
PRICE_OUTPUT_PER_MTOK = 15.00

# Conversation memory mode
MEMORY_MODE = "FULL"    # "FULL" | "ROLLING" | "STATELESS"
MEMORY_WINDOW = 10

# Whether to store full raw API responses in JSONL
SAVE_RAW_JSON = True

# Temperature settings
TEMPERATURES: Dict[str, float] = {
    #"ClueTemperature_0_4": 0.4,
    #"ClueTemperature_0_5": 0.5,
    #"ClueTemperature_0_6": 0.6,
    #"ClueTemperature_0_7": 0.7,
    #"ClueTemperature_0_8": 0.8,
    #"ClueTemperature_0_9": 0.9,
    "ClueTemperature_1_0": 1.0,
}

# File paths
INPUT_FILE = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\bga_data_en.csv")
OUTPUT_FILE = Path(r"C:\Users\serge\Documents\Thesis\Data Seafile\Output\Output\just_one_clues_claude_temp_test_2.csv")
RAW_JSONL_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_raw.jsonl")

# Runtime parameters
MAX_ROWS = None
CHECKPOINT_EVERY = 25
REQUEST_DELAY_SEC = 0.4
MAX_RETRIES = 4
BACKOFF_BASE = 1.8
TOKEN_WARNING = 100_000
TOKEN_LIMIT = 200_000   # just for your warning message

# System prompt
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
USER_PROMPT_NEXT = 'The next Mystery word is: "{mystery}". Respond with one valid one-word clue.'


# =========================
# Helper functions
# =========================

def backoff_sleep(attempt: int):
    """Wait longer after each failed attempt."""
    time.sleep((BACKOFF_BASE ** attempt) + (0.05 * attempt))


def trim_messages_for_mode(messages: List[Dict], mode: str, window_size: int = 10) -> List[Dict]:
    """
    Keep or reduce chat history depending on memory mode.

    NOTE (Claude): Anthropic takes `system=` separately. We still keep the system
    message in our in-memory list for compatibility, but we will strip it before sending.
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


def to_anthropic_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Split out `system` and return (system_text, messages_without_system)
    Anthropic expects:
      - system: str | list[content blocks]
      - messages: list[{role: "user"|"assistant", content: list[blocks]}]
    """
    system_text = ""
    stripped: List[Dict] = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_text = content or ""
            continue
        if role in ("user", "assistant"):
            stripped.append({
                "role": role,
                "content": [{"type": "text", "text": content or ""}],
            })
    return system_text, stripped


def chat_once(messages: List[Dict], temperature: float):
    # Prepare Anthropic-style messages
    system_text, conv = to_anthropic_messages(messages)

    # Cache the system prompt for 1 hour
    system_blocks = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]

    # Send request to Claude
    resp = client.messages.create(
        model=MODEL,
        max_tokens=15,
        temperature=temperature,
        system=system_blocks,
        messages=conv,
    )

    # Extract first text block
    text = ""
    if resp.content:
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                text = (blk.text or "").strip()
                break

    # Parse usage
    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "total_tokens": (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0)
        ) if usage else 0,
    }

    # print cache stats (shows live cache hits/writes)
    if usage:
        c_create = getattr(usage, "cache_creation_input_tokens", 0)
        c_read = getattr(usage, "cache_read_input_tokens", 0)
        print(
            f"   Cache → created: {c_create} | read: {c_read} "
            f"| input_tokens: {usage_dict['input_tokens']} | output_tokens: {usage_dict['output_tokens']}"
        )

    return text, usage_dict, resp




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

    # Robustly serialize Anthropic response
    response_payload = None
    try:
        if hasattr(full_response, "model_dump_json"):
            response_payload = json.loads(full_response.model_dump_json())
        elif hasattr(full_response, "model_dump"):
            response_payload = full_response.model_dump()
        else:
            response_payload = json.loads(json.dumps(full_response, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        response_payload = str(full_response)

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
    """ Save a run summary to a standalone JSON file (next to the output CSV). Always runs at the end. """
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
            raw, usage_dict, full_response = chat_once(call_messages, temperature)
            messages.append({"role": "assistant", "content": raw})
            return raw, usage_dict, messages, full_response
        except Exception as e:
            attempt += 1
            print(f"[Anthropic ERROR attempt {attempt}] {type(e).__name__}: {e}")  # <--- add this
            backoff_sleep(attempt)


    messages.append({"role": "assistant", "content": "ERROR"})
    return "ERROR", last_usage, messages, None


# =========================
# Main logic
# =========================

def main():
    print("Loading input file ...")
    started_at_iso = datetime.now().isoformat(timespec="seconds")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8", nrows=20)
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

            # local conversation memory; system kept for bookkeeping
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

                if tot_t > TOKEN_WARNING:
                    print(f"Warning: context size {tot_t} tokens (limit ~{TOKEN_LIMIT})")

                print(f" Raw answer: {raw!r} | Final clue saved: {candidate!r} | Tokens in/out: {in_t}/{out_t}")

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

        # Save standalone JSON summary
        ended_at_iso = datetime.now().isoformat(timespec="seconds")
        save_run_summary_json(
            grand_in_tokens,
            grand_out_tokens,
            started_at_iso=started_at_iso,
            ended_at_iso=ended_at_iso,  # typo fix in next line
        )

        # Console summary
        total_in_usd = (grand_in_tokens / 1_000_000) * PRICE_INPUT_PER_MTOK
        total_out_usd = (grand_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK

        print("\n===== Run Summary =====")
        print(f"Total tokens input/output: {grand_in_tokens}/{grand_out_tokens}")
        print(f"Costs (approx.): ${total_in_usd:.4f} in + ${total_out_usd:.4f} out = ${total_in_usd + total_out_usd:.4f}")
        print("=======================")


if __name__ == "__main__":
    main()
