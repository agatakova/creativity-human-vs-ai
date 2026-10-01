# The Creativity Contest: Creative Idea Generation in Humans and AI Chatbots

Code for my Master's thesis (M.Sc. Quantitative Decision Making in Economics and Management, Johannes Gutenberg University Mainz, 2026).

**Author:** Agata / serge

## Overview

The thesis compares the creativity of humans and large language models in the cooperative word game *Just One*. In this game, players give one-word clues that help a teammate guess a hidden *mystery word*. Identical clues cancel each other out, so good clues have to be both helpful and original.

The project has two parts:

1. **Clue originality:** Four AI systems (GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro and DeepSeek-Chat) generate clues for the same mystery words that human players received. The originality of every clue is measured and compared between humans and AI models, and across temperature and memory settings.
2. **Guessing experiment:** GPT-4o acts as the guessing player and tries to identify the mystery word from bundles of human or AI clues. This tests whether more original clues are still useful.

## Pipeline

```
Human game data (Board Game Arena)
        │
        ├──► 1_clue_generation        AI clues via the APIs of four LLMs
        │            │
        │            ▼
        ├──► 2_clue_originality       data reshaping, validation, originality index
        │            │
        │            ▼
        ├──► 3_guessing_experiment    human clue sample, GPT-4o guesses, merging
        │            │
        │            ▼
        └──► 4_analysis               descriptive statistics, regressions, plots
```

## Repository structure

| Folder / script | Purpose |
|---|---|
| **1_clue_generation/** | |
| `just_one_clues_chatgpt6_API.py` | Generates clues with GPT-4o (OpenAI API) |
| `just_one_clues_claude3_API.py` | Generates clues with Claude Sonnet 4.5 (Anthropic API) |
| `just_one_clues_gemini5_API.py` | Generates clues with Gemini 2.5 Pro (Google Gen AI API) |
| `just_one_clues_deepseek5_API.py` | Generates clues with DeepSeek-Chat (DeepSeek API) |
| **2_clue_originality/** | |
| `explode_human_clues.py` | Parses the human clue lists and reshapes them to one row per clue |
| `originality_high_temp_all_models.py` | Computes originality for the AI clues of all four models (wide format) |
| `just_one_merged_data_with_originality.py` | Cleans and validates all human and AI clues and computes the originality index (long format) |
| **3_guessing_experiment/** | |
| `just_one_clues_humans_with_originality.py` | Draws a reproducible random sample of four human clues per mystery word (seed 42) |
| `just_one_clues_chatgpt_guesses4.py` | GPT-4o guesses the mystery word from human or AI clue bundles (up to three attempts) |
| `merging_guessing_data.py` | Merges and cleans the results of the guessing runs |
| `originality_guesses_mergeg.py` | Adds the originality index to the merged guessing data |
| **4_analysis/** | |
| `Analysen.py` | Clue originality: descriptive statistics, t-tests, OLS with robust and clustered standard errors, density, violin and box plots |
| `Analysen_guesses3.py` | Guessing experiment: aggregation per mystery word, paired t-tests, success probabilities, plots |

## Methods and technical features

**Data collection via APIs** (folder 1 and the guessing script)
- Loops over all mystery words and configurable temperature settings
- Three conversation memory modes: `FULL`, `ROLLING` and `STATELESS`
- Retries with exponential backoff for failed API calls
- Checkpoints every 20 requests and automatic resumption of interrupted runs (already processed words are skipped)
- Raw API responses logged as JSONL, plus a JSON run summary with token usage and estimated cost
- Automatic flagging of empty or invalid model outputs

**Data validation and preparation**
- Normalisation of clues (lowercase, punctuation removed)
- Regex-based detection of malformed or non-Latin tokens
- Reshaping of list-type clue columns into long format, merging of human and AI data

**Originality index**
- Originality = cosine distance between the clue and the mystery word × 100, based on pretrained [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) English word vectors (`cc.en.300.bin`)
- A higher value means the clue is semantically further away from the mystery word
- Word vectors are cached to speed up processing

**Analysis**
- OLS regressions (statsmodels) with heteroskedasticity-robust and clustered standard errors
- t-tests and paired t-tests with confidence intervals
- Visualisation with Seaborn and Matplotlib

## Data

The datasets are **not included** in this repository.

- **Human data:** Game data from *Just One* on [Board Game Arena](https://boardgamearena.com), collected over one year by my supervising chair at JGU Mainz. The data is not mine to publish. The scripts expect a CSV file (`bga_data_en.csv`) with one row per mystery word, a column `Mystery Word` and a column `Clues` containing a list of the human clues, e.g. `"['inflection', 'twang', 'dialect']"`.
- **AI data:** Generated with the scripts in folder 1.
- **Manual steps:** Combining the outputs of the four models into one file (`just_one_clues_hightemp_allmodels.csv`) and creating the long-format file of human and AI clues (`data_merged_clues_ai.csv`) were done manually in Excel or with short ad hoc code that is not part of this repository.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Set the API keys as environment variables (keys are never stored in the code):
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `DEEPSEEK_API_KEY`
3. Download the fastText English model `cc.en.300.bin` from [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html).
4. Adjust the file paths at the top of each script. The paths are absolute Windows paths from my local machine.

## Notes

- This repository contains the code as used for the submitted thesis. Apart from the folder structure, it has not been modified afterwards.
- Some comments in the code are in German.
- The settings in each script (model, temperatures, memory mode, file names) reflect the last run. Other configurations from the thesis can be reproduced by changing these settings.
