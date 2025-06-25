#!/usr/bin/env python3
"""
merge_resolved_relations.py

Two-pass LLM-based canonicalization of legal relation variants.

Phase 1:
- Scans each subfolder under INPUT_DIR for "results/resolved_<folder>.csv"
- Aggregates all "resolved_relation" variants
- Splits variants into chunks and sends each chunk to the LLM for canonicalization
- Merges all chunk results into a flat list of canonical mappings

Phase 2:
- Splits the first-pass canonical labels into chunks and sends each to the LLM
- Merges chunk results into a final mapping and aggregates document counts
- Outputs:
    • combined_relations.csv        — all distinct variants and their document counts
    • combined_relations_list.txt  — bullet list (for reference)
    • first_pass_relations.csv     — mapping after first-pass canonicalization with document-based counts
    • llm_resolved_relations.json  — merged LLM output mapping
    • merged_relations.csv         — final mapping of each canonical group to summed document counts
"""

import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / 'entity_types'))

from entity_relation_extraction import call_deepseek, load_prompt_template

INPUT_DIR = Path("output/relations")
PROMPT_PATH = Path("prompts/relation_postprocessing.txt")
COMBINED_CSV = INPUT_DIR / "combined_relations.csv"
COMBINED_LIST = INPUT_DIR / "combined_relations_list.txt"
FIRST_PASS_CSV = INPUT_DIR / "first_pass_relations.csv"
LLM_JSON = INPUT_DIR / "llm_resolved_relations.json"
MERGED_CSV = INPUT_DIR / "merged_relations.csv"
CHUNK_SIZE = 100
MAX_LLM_RETRIES = 3


def soft_normalize(text: str) -> str:
    """Lowercase, preserve word boundaries as underscores."""
    return re.sub(r'\s+', '_', text.strip().lower())


def hard_normalize(text: str) -> str:
    """Lowercase, preserve only a-z, 0-9, and underscores."""
    return re.sub(r'[^a-z0-9_]', '', soft_normalize(text))


def extract_json_object(text):
    brace_count = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if start is None:
                start = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start is not None:
                return text[start:i+1]
    return ""


def main():
    prompt_template = load_prompt_template(str(PROMPT_PATH))
    if '{relation_list}' not in prompt_template:
        print("Prompt must contain '{relation_list}' placeholder")
        sys.exit(1)

    # Step 1: Aggregate raw variants by document
    raw_counter = Counter()
    for sub in tqdm(INPUT_DIR.iterdir(), desc="Scanning folders"):
        if not sub.is_dir():
            continue
        csv_file = sub / 'results' / f"resolved_{sub.name}.csv"
        if not csv_file.exists():
            continue
        seen = set()
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    norm = soft_normalize(row[0].strip())
                    if norm:
                        seen.add(norm)
        for norm in seen:
            raw_counter[norm] += 1

    variants = sorted(raw_counter.keys())
    with open(COMBINED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variant','doc_count'])
        for v in variants:
            writer.writerow([v, raw_counter[v]])
    with open(COMBINED_LIST, 'w', encoding='utf-8') as f:
        for v in variants:
            f.write(f"- {v}\n")

    # Step 2: First-pass canonicalization with document-based counts
    first_pass_mapping = defaultdict(list)
    canon_counter = Counter()
    for i in tqdm(range(0, len(variants), CHUNK_SIZE), desc="First-pass canonicalization"):
        chunk = variants[i:i+CHUNK_SIZE]
        bullets = '\n'.join(f"- {v}" for v in chunk)
        prompt = prompt_template.format(relation_list=bullets)
        for attempt in range(1, MAX_LLM_RETRIES+1):
            out = call_deepseek(prompt)
            text = out[1] if isinstance(out, (list,tuple)) and isinstance(out[1],str) else str(out)
            snippet = extract_json_object(text)
            try:
                mapping = json.loads(snippet)
                for canon, group in mapping.items():
                    for v in group:
                        if isinstance(v, str):
                            norm = soft_normalize(v)
                            first_pass_mapping[canon].append(norm)
                            canon_counter[canon] += raw_counter.get(norm, 0)
                break
            except Exception:
                if attempt == MAX_LLM_RETRIES:
                    print("LLM failed on chunk", i)
                    sys.exit(1)

    with open(FIRST_PASS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical','doc_count','variants'])
        for canon, group in first_pass_mapping.items():
            writer.writerow([canon, canon_counter[canon], ', '.join(group)])

    # Step 3: Global canonicalization with chunking
    global_inputs = sorted(first_pass_mapping.keys())
    full_mapping = defaultdict(list)
    merged_counter = Counter()

    for i in tqdm(range(0, len(global_inputs), CHUNK_SIZE), desc="Global canonicalization"):
        chunk = global_inputs[i:i+CHUNK_SIZE]
        bullets = '\n'.join(f"- {c}" for c in chunk)
        prompt = prompt_template.format(relation_list=bullets)
        print(prompt)
        for attempt in range(1, MAX_LLM_RETRIES+1):
            out = call_deepseek(prompt)
            print(out)
            text = out[1] if isinstance(out, (list,tuple)) and isinstance(out[1],str) else str(out)
            snippet = extract_json_object(text)
            try:
                chunk_map = json.loads(snippet)
                for new_canon, group in chunk_map.items():
                    for old in group:
                        if isinstance(old, str):
                            full_mapping[new_canon].append(old)
                break
            except Exception:
                if attempt == MAX_LLM_RETRIES:
                    print("LLM failed on chunk", i)
                    sys.exit(1)

    # Aggregate merged_counter from full_mapping
    for canon, olds in full_mapping.items():
        merged_counter[canon] = sum(canon_counter.get(old, 0) for old in olds)

    # Save raw LLM mapping
    with open(LLM_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_mapping, f, indent=2, ensure_ascii=False)

    # Step 4: Final merging - write merged CSV
    with open(MERGED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical','total_doc_count','variants'])
        for canon, olds in full_mapping.items():
            key = hard_normalize(canon)
            writer.writerow([key, merged_counter.get(canon, 0), ', '.join(olds)])

if __name__ == '__main__':
    main()
