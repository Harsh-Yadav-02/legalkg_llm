#!/usr/bin/env python3
"""
canonicalize_relations_pipeline.py

Postprocess folder-wise raw relation-lists from CSV to canonicalized mappings via LLM,
then save both JSON and CSV outputs.

For each subfolder under INPUT_DIR:
 1. Look for "results/<FolderName>.csv".
 2. Read column 2, deduplicated raw relation-lists (string blocks).
 3. Build a bullet-list block of each folder’s relations.
 4. Call Deepseek with the prompt-template to canonicalize them into a JSON mapping
    (canonical_relation → [original_variants]).
 5. Write:
    • "resolved_<FolderName>.json": the mapping object.
    • "resolved_<FolderName>.csv": two columns: resolved_relation, merged_variants.

Usage:
    python canonicalize_relations_pipeline.py \
      --input_dir output/new_27 \
      --prompt_template prompts/relation_postprocessing.txt \
      --num_workers 8
"""
import argparse
import csv
import json
import sys
import multiprocessing
import time
import re
from pathlib import Path
from tqdm import tqdm

# Insert path to entity_relation_extraction.py
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / 'entity_types'))

from entity_relation_extraction import (
    call_deepseek,
    load_prompt_template
)

############################
# CONFIGURATION
############################
DEFAULT_INPUT_DIR = Path("output/relations")
DEFAULT_PROMPT_TEMPLATE_PATH = Path("prompts/relation_postprocessing.txt")
MAX_RETRIES = 10
NUM_WORKERS = multiprocessing.cpu_count()

############################
# CSV I/O HELPERS
############################

def load_second_column(csv_path: Path):
    """Read unique, non-empty values from column 2."""
    seen = set()
    vals = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) > 1:
                v = row[1].strip()
                if v and v not in seen:
                    seen.add(v)
                    vals.append(v)
    return vals


def write_mapping_json(results_folder: Path, folder_name: str, mapping: dict):
    out = results_folder / f"resolved_{folder_name}.json"
    with open(out, 'w', encoding='utf-8') as jf:
        json.dump(mapping, jf, indent=2, ensure_ascii=False)


def write_mapping_csv(results_folder: Path, folder_name: str, mapping: dict):
    out = results_folder / f"resolved_{folder_name}.csv"
    with open(out, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["resolved_relation", "merged_variants"])
        for canon, variants in mapping.items():
            writer.writerow([canon, ", ".join(variants)])

############################
# PROCESSING
############################

def process_subfolder(args_tuple):
    subfolder, prompt_template = args_tuple
    name = subfolder.name
    results = subfolder / 'results'
    csv_file = results / f"{name}.csv"
    if not csv_file.exists():
        return (name, False, "No results CSV found.")

    raw_list = load_second_column(csv_file)
    if not raw_list:
        return (name, False, "Empty column 2 list.")

    bullets = "\n".join(f"- {item}" for item in raw_list)
    prompt = prompt_template.format(relation_list=bullets)

    mapping = None
    for attempt in range(1, MAX_RETRIES + 1):
        out = call_deepseek(prompt)
        # Debug print
        print(f"Out->>>>>>>>>>>>>> {out}")

        # Unpack LLM output: prefer grabbing the JSON-containing str
        if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], str):
            text = out[1]
        elif isinstance(out, (tuple, list)) and isinstance(out[0], str):
            text = out[0]
        elif hasattr(out, 'text'):
            text = out.text
        elif isinstance(out, dict) and 'choices' in out:
            text = out['choices'][0].get('text', '')
        else:
            text = str(out)

        # Extract first JSON object
        match = re.search(r'\{[\s\S]*?\}', text, flags=re.DOTALL)
        if not match:
            snippet = ''
        else:
            snippet = match.group(0)

        # Debug print
        print(f"Extracted JSON snippet (attempt {attempt}): {snippet}")

        # Try parse
        try:
            mapping = json.loads(snippet)
            if not isinstance(mapping, dict):
                raise ValueError("Parsed JSON is not a dict")
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            else:
                return (name, False, f"Failed parsing JSON after {MAX_RETRIES} retries. Last error: {e}")

    # Write outputs
    try:
        write_mapping_json(results, name, mapping)
        write_mapping_csv(results, name, mapping)
    except Exception as e:
        return (name, False, f"Write error: {e}")

    return (name, True, f"Wrote {len(mapping)} canonical relations.")

############################
# MAIN
############################

def main():
    p = argparse.ArgumentParser("Canonicalize relation lists via LLM")
    p.add_argument('--input_dir', type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument('--prompt_template', type=Path, default=DEFAULT_PROMPT_TEMPLATE_PATH)
    p.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    args = p.parse_args()

    if not args.input_dir.is_dir():
        print("Invalid input_dir", file=sys.stderr)
        sys.exit(1)

    prompt_t = load_prompt_template(str(args.prompt_template))
    if prompt_t is None:
        print("Cannot load prompt_template", file=sys.stderr)
        sys.exit(1)

    subs = [d for d in sorted(args.input_dir.iterdir()) if d.is_dir()]
    tasks = [(d, prompt_t) for d in subs]

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_subfolder, tasks), total=len(tasks)))

    for name, ok, msg in results:
        sym = '✔' if ok else '✖'
        tqdm.write(f"{sym} {name}: {msg}")

    print("Done.")

if __name__ == '__main__':
    main()
