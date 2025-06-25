#!/usr/bin/env python3
"""
merge_resolved_entities.py

Script to merge entity type variants into normalized canonical types using LLM assistance.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# HARDCODED PATHS
INPUT_DIR = Path("output/new_27")
PROMPT_PATH = Path("prompts/etype_postprocessing2.txt")

# Derived output paths:
COMBINED_CSV = INPUT_DIR / "combined_variants.csv"
COMBINED_LIST_TXT = INPUT_DIR / "combined_variants_list.txt"
OUTPUT_MERGED_CSV = INPUT_DIR / "merged_resolved.csv"
INTERMEDIATE_JSON = INPUT_DIR / "llm_resolved_types.json"

MAX_LLM_RETRIES = 3

from entity_relation_extraction import (
    call_deepseek,
    clean_output,
    extract_json,
    load_prompt_template
)

def soft_normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip()).lower()

def hard_normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())

def load_first_column(csv_path: Path):
    seen = set()
    ordered = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _header = next(reader, None)
        for row in reader:
            if len(row) > 1:
                val = row[0].strip()
                norm_val = soft_normalize(val)
                if norm_val and norm_val not in seen:
                    seen.add(norm_val)
                    ordered.append(val)  # use original value for better prompt
    return ordered

def build_bullet_list(entity_types_list):
    normalized_seen = set()
    normalized_ordered = []
    for raw in entity_types_list:
        norm = soft_normalize(raw)
        if norm and norm not in normalized_seen:
            normalized_seen.add(norm)
            normalized_ordered.append(raw)  # keep original form for readability
    return "\n".join(f"- {etype}" for etype in normalized_ordered)

def parse_llm_json_with_retry(raw_output: str, max_retries: int = 1):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        cleaned = clean_output(raw_output)
        json_snippet = extract_json(cleaned).strip()
        try:
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

            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object. Got: {type(parsed)}")
            for k, v in parsed.items():
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError(f"Expected list of strings for key '{k}', got: {v}")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_exception = e
            if attempt < max_retries:
                continue
            else:
                raise
    raise last_exception

def main():
    if not INPUT_DIR.is_dir():
        print(f"ERROR: INPUT_DIR '{INPUT_DIR}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if not PROMPT_PATH.exists() or not PROMPT_PATH.is_file():
        print(f"ERROR: Prompt template '{PROMPT_PATH}' not found.", file=sys.stderr)
        sys.exit(1)

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    if "{entity_list}" not in prompt_template:
        print("ERROR: Prompt template must contain the placeholder '{entity_list}'.", file=sys.stderr)
        sys.exit(1)

    variant_counter = Counter()
    for subfolder in tqdm(list(INPUT_DIR.iterdir()), desc="Scanning folders"):
        if not subfolder.is_dir():
            continue

        resolved_csv = subfolder / "results" / f"resolved_{subfolder.name}.csv"
        if not resolved_csv.exists():
            tqdm.write(f"→ Skipping '{subfolder.name}': {resolved_csv.name} not found.")
            continue

        with open(resolved_csv, newline="", encoding="utf-8") as cf:
            reader = csv.reader(cf)
            next(reader, None)
            for row in reader:
                if len(row) > 1:
                    variant = row[0].strip()
                    norm_variant = soft_normalize(variant)
                    if norm_variant:
                        variant_counter[norm_variant] += 1

    if not variant_counter:
        print("No variants found across any resolved CSV. Exiting.")
        sys.exit(0)

    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["variant", "count"])
        for variant, cnt in sorted(variant_counter.items()):
            writer.writerow([variant, cnt])
    print(f"📂 Combined variants CSV written to: {COMBINED_CSV.name}")

    with open(COMBINED_LIST_TXT, "w", encoding="utf-8") as fout:
        for variant in sorted(variant_counter.keys()):
            fout.write(f"- {variant}\n")
    print(f"📄 Combined variants list written to: {COMBINED_LIST_TXT.name}")

    raw_entity_list = load_first_column(COMBINED_CSV)
    bullets = build_bullet_list(raw_entity_list)

    try:
        prompt_filled = prompt_template.format(entity_list=bullets)
    except KeyError as ke:
        print(f"ERROR: Failed to format prompt: {ke}", file=sys.stderr)
        sys.exit(1)

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        print(f"🔁 LLM attempt {attempt}/{MAX_LLM_RETRIES}")
        try:
            json_snippet, _ = call_deepseek(prompt_filled)
            if not json_snippet:
                raise ValueError("LLM returned no JSON snippet.")
            llm_mapping = parse_llm_json_with_retry(json_snippet, max_retries=1)
            break
        except Exception as e:
            print(f"⚠️  Attempt {attempt} failed: {e}")
            if attempt == MAX_LLM_RETRIES:
                print("❌ Exceeded max retries. Exiting.")
                sys.exit(1)

    with open(INTERMEDIATE_JSON, "w", encoding="utf-8") as jf:
        json.dump(llm_mapping, jf, indent=2, ensure_ascii=False)
    print(f"📦 Canonical mapping saved to: {INTERMEDIATE_JSON.name}")

    # Build mapping: variant → canonical type
    variant_to_canon = {}
    for canon_type, variants in llm_mapping.items():
        canon_norm = hard_normalize(canon_type)
        for v in variants:
            variant_norm = soft_normalize(v)
            variant_to_canon[variant_norm] = canon_norm

    # Reverse mapping: canonical type → list of variants
    canon_to_variants = {}
    for variant, canon in variant_to_canon.items():
        canon_to_variants.setdefault(canon, []).append(variant)

    # Count total variants per canonical type
    canon_totals = Counter()
    for variant, cnt in variant_counter.items():
        canon = variant_to_canon.get(variant, hard_normalize(variant))
        canon_totals[canon] += cnt

    with open(OUTPUT_MERGED_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["variant", "count", "resolved_type", "merged_total_count", "merged_variants"])
        for variant, cnt in tqdm(sorted(variant_counter.items()), desc="Writing merged CSV"):
            canon = variant_to_canon.get(variant, variant)
            canon_hard = hard_normalize(canon)
            merged_total = canon_totals[canon_hard]
            merged_list = ", ".join(sorted(canon_to_variants.get(canon_hard, [])))
            writer.writerow([variant, cnt, canon_hard, merged_total, merged_list])
    print(f"✅ Merged CSV written to: {OUTPUT_MERGED_CSV.name}")

if __name__ == "__main__":
    main()
