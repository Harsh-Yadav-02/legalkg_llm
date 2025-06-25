#!/usr/bin/env python3
"""
etype_postprocessing_pipeline.py

This version hard‐codes the input/output paths and prompt‐template path at the top of the script,
instead of accepting them via CLI arguments. A tqdm progress bar is used to show overall folder‐wise progress.
We've also added:
  - Retry logic (up to 3 attempts) if JSON decode/parsing fails.
  - A --num_workers argument to parallelize folder processing.

For each subfolder under INPUT_DIR:
 1. Looks for "results/<FolderName>.csv" inside that subfolder.
 2. Reads column 2 (entity types), deduplicates them, normalizes each by lowercasing and replacing
    any non-alphanumeric character with a space (e.g. "Learned_counsel" → "learned counsel"), and then builds a bullet-list block.
 3. Loads the prompt template (which must contain a literal "{entity_list}" placeholder).
 4. Calls Deepseek via the existing call_deepseek() from entity_relation_extraction.py.
 5. Parses the returned JSON mapping (canonical_type → [original_variants]), automatically fixing
    a common case where the model emits a standalone JSON array instead of a key:value pair.
    If a parse error occurs, it retries up to 3 times.
 6. Writes out two files inside the results folder:
      • "resolved_<FolderName>.json" containing the mapping as JSON.
      • "resolved_<FolderName>.csv" with columns: resolved_type, merged_types.

To run, simply edit the hard‐coded paths below (or pass them via CLI) and execute:
    python etype_postprocessing_pipeline.py --input_dir output/new_27 --prompt_template prompts/etype_postprocessing.txt --num_workers 8
"""

import argparse
import csv
import json
import re
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Import the LLM‐calling utilities from your existing file.
# Make sure entity_relation_extraction.py is in the same directory or on your PYTHONPATH.
from entity_relation_extraction import (
    call_deepseek,
    clean_output,
    extract_json,
    load_prompt_template
)

############################
# 1. HARD‐CODED (DEFAULT) CONFIGURATION
############################

DEFAULT_INPUT_DIR = Path("output/new_27")         # ← EDIT or override via --input_dir
DEFAULT_PROMPT_TEMPLATE_PATH = Path("prompts/etype_postprocessing.txt")  # ← EDIT or override

#########################
# 2. UTILITY FUNCTIONS
#########################

def normalize_type(text: str) -> str:
    """
    Replace any non-alphanumeric character with a space, collapse multiple spaces into one,
    strip leading/trailing spaces, and lowercase the result.
    E.g. "Learned_counsel" → "learned counsel"
    """
    replaced = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    collapsed = re.sub(r'\s+', ' ', replaced).strip().lower()
    return collapsed

def load_second_column(csv_path: Path):
    """
    Read the CSV at csv_path and return a deduplicated list of raw values
    found in the second column (index 1), preserving insertion order.
    """
    seen = set()
    ordered = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) > 1:
                val = row[1].strip()
                if val and val not in seen:
                    seen.add(val)
                    ordered.append(val)
    return ordered

def build_bullet_list(entity_types_list):
    """
    Given a list of entity-type strings, normalize each (lowercase & replace special chars),
    deduplicate again (in case normalization collapsed different raw values to the same string),
    and return a bullet-list block:
      "- typea\n- typeb\n- typec\n"
    """
    normalized_seen = set()
    normalized_ordered = []
    for raw in entity_types_list:
        norm = normalize_type(raw)
        if norm and norm not in normalized_seen:
            normalized_seen.add(norm)
            normalized_ordered.append(norm)

    return "\n".join(f"- {etype}" for etype in normalized_ordered)

def parse_llm_json_with_retry(raw_output: str, max_retries: int = 1):
    """
    Try parsing the LLM response (raw_output) into a dict, with up to `max_retries` attempts
    if ValueError or JSONDecodeError is raised. Returns the parsed dict on success,
    or re-raises the last exception if all attempts fail.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        cleaned = clean_output(raw_output)
        json_snippet = extract_json(cleaned).strip()

        try:
            # If snippet does not start with '{', try to fix common case:
            if not json_snippet.startswith("{"):
                match = re.match(r'^\s*\[\s*"([^"]+)"\s*\]\s*,?', json_snippet)
                if match:
                    val = match.group(1)
                    fixed_start = f'"{val}": ["{val}"],'
                    json_snippet = re.sub(r'^\s*\[\s*"[^"]+"\s*\]\s*,?', fixed_start, json_snippet, count=1)
                inner = json_snippet.rstrip(",\n\r\t ")
                wrapped = "{" + inner + "}"
                parsed = json.loads(wrapped)
            else:
                parsed = json.loads(json_snippet)

            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object mapping. Got: {type(parsed)}")
            for k, v in parsed.items():
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError(f"Expected each value to be a list of strings. Key '{k}' has: {v}")

            return parsed  # success

        except (json.JSONDecodeError, ValueError) as e:
            last_exception = e
            if attempt < max_retries:
                continue  # retry
            else:
                # after final attempt, re-raise
                raise

    # If somehow we exit the loop without returning, re-raise
    raise last_exception

def write_resolved_csv(results_folder: Path, mapping: dict):
    """
    Write out results_folder/resolved_<FolderName>.csv with columns:
      resolved_type, merged_types (comma-joined).
    """
    folder_name = results_folder.parent.name
    out_name = f"resolved_{folder_name}.csv"
    out_path = results_folder / out_name
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["resolved_type", "merged_types"])
        for resolved, variants in mapping.items():
            merged = ", ".join(variants)
            writer.writerow([resolved, merged])

def write_resolved_json(results_folder: Path, mapping: dict):
    """
    Write out results_folder/resolved_<FolderName>.json containing the mapping in JSON form.
    """
    folder_name = results_folder.parent.name
    out_name = f"resolved_{folder_name}.json"
    out_path = results_folder / out_name
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(mapping, jf, indent=2, ensure_ascii=False)

#########################
# 3. PER-FOLDER PROCESSING
#########################

def process_subfolder(args_tuple):
    """
    Process exactly one <INPUT_DIR>/<subfolder>:
      - Read CSV in subfolder/results/<subfolder>.csv
      - Build bullet list
      - Fill prompt, call LLM (with retry)
      - Parse JSON, write out resolved JSON+CSV
    Returns a tuple: (subfolder_name, success_flag, message)
    """
    subfolder, raw_template = args_tuple
    folder_name = subfolder.name
    results_folder = subfolder / "results"
    csv_filename = f"{folder_name}.csv"
    csv_path = results_folder / csv_filename

    if not results_folder.is_dir() or not csv_path.exists():
        return (folder_name, False, f"Skipping '{folder_name}': no 'results/{csv_filename}' found.")

    raw_entity_list = load_second_column(csv_path)
    if not raw_entity_list:
        return (folder_name, False, f"Skipping '{folder_name}': no non-empty values in column 2.")

    bullets = build_bullet_list(raw_entity_list)

    try:
        prompt_filled = raw_template.format(entity_list=bullets)
    except KeyError as ke:
        return (folder_name, False,
                f"ERROR: Prompt template missing '{{entity_list}}'. KeyError: {ke}. Skipping.")

    # Attempt up to 3 times if JSON parsing fails
    for attempt in range(1, 10):
        try:
            json_snippet, _ = call_deepseek(prompt_filled)
            if not json_snippet:
                return (folder_name, False, f"LLM returned no JSON snippet for '{folder_name}'.")
            mapping = parse_llm_json_with_retry(json_snippet, max_retries=1)
            # If parse succeeds, break out of retry loop:
            break
        except Exception as e:
            if attempt < 3:
                continue
            else:
                return (folder_name, False,
                        f"ERROR: Failed to parse LLM JSON for '{folder_name}' after 10 attempts: {e}")

    # Write out files
    try:
        write_resolved_json(results_folder, mapping)
        write_resolved_csv(results_folder, mapping)
    except Exception as e:
        return (folder_name, False, f"ERROR writing output for '{folder_name}': {e}")

    return (folder_name, True,
            f"Wrote '{results_folder / f'resolved_{folder_name}.json'}' and '{results_folder / f'resolved_{folder_name}.csv'}'.")

#########################
# 4. MAIN LOGIC
#########################

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess entity‐type normalization via Deepseek, with retry logic and multiprocessing."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Path to the parent directory containing subfolders (each with a 'results/<FolderName>.csv')."
    )
    parser.add_argument(
        "--prompt_template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE_PATH,
        help="Path to the prompt template file (must contain '{entity_list}')."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers to process subfolders (default: all CPU cores)."
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    PROMPT_TEMPLATE_PATH = args.prompt_template
    NUM_WORKERS = args.num_workers

    # Validate paths
    if not INPUT_DIR.is_dir():
        print(f"ERROR: INPUT_DIR '{INPUT_DIR}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not PROMPT_TEMPLATE_PATH.exists():
        print(f"ERROR: Prompt template '{PROMPT_TEMPLATE_PATH}' not found.", file=sys.stderr)
        sys.exit(1)

    raw_template = load_prompt_template(str(PROMPT_TEMPLATE_PATH))
    if raw_template is None:
        print(f"ERROR: Failed to load prompt template '{PROMPT_TEMPLATE_PATH}'.", file=sys.stderr)
        sys.exit(1)

    # Collect all valid subfolders
    all_subfolders = [sf for sf in sorted(INPUT_DIR.iterdir()) if sf.is_dir()]

    # Prepare arguments for pool
    pool_args = [(sf, raw_template) for sf in all_subfolders]

    # Process in parallel
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_subfolder, pool_args),
                            total=len(pool_args),
                            desc="Processing folders",
                            unit="folder"))

    # Print summary
    for folder_name, success, message in results:
        prefix = "✔" if success else "→"
        tqdm.write(f"{prefix} {folder_name}: {message}")

    print("Done processing all folders.")

if __name__ == "__main__":
    main()
