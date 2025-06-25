#!/usr/bin/env python3
"""
define_entity_types.py

Reads a list of entity types from a hard‐coded text file (one term per line),
prompts the LLM (deepseek-r1:14b via Ollama) to generate a concise legal definition for each,
and writes the results to a .txt file in the format “entity_type: definition” (one per line).

If parsing the JSON fails, it will retry up to 10 times for that entity before giving up.

All paths are defined at the top of this file.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# ──────── HARDCODED PATHS ────────────────────────────────────────────────────────
ENTITY_FILE_PATH      = Path("datasets/etypes_100.txt")
PROMPT_TEMPLATE_PATH  = Path("prompts/top100etype_definition.txt")
OUTPUT_TXT_PATH       = Path("datasets/entity_types_100.txt")
# ────────────────────────────────────────────────────────────────────────────────

# Import utility functions from your existing entity_relation_extraction module:
from entity_relation_extraction import (
    load_prompt_template,   # reads a prompt template and checks its placeholder
    call_deepseek,          # invokes `ollama run deepseek-r1:14b`
    clean_output            # strips <think>…</think> and collapses whitespace
)

def load_entity_list(file_path: Path):
    """
    Load one entity type per line from a plain text file.
    Ignores blank lines and strips whitespace.
    """
    if not file_path.is_file():
        print(f"ERROR: Entity file not found at: {file_path}", file=sys.stderr)
        sys.exit(1)

    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            term = line.strip()
            if term:
                entities.append(term)
    return entities

def extract_json_object(text: str) -> str:
    """
    Find the first JSON object (“{ … }”) in `text` and return it.
    If none found, return an empty string.
    """
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    return match.group(0) if match else ""

def main():
    # 1) Load the list of entity types (one per line).
    entities = load_entity_list(ENTITY_FILE_PATH)
    if not entities:
        print("No entity types found; nothing to do.", file=sys.stderr)
        sys.exit(0)

    # 2) Load the prompt template (must contain "{entity_type}" exactly once).
    if not PROMPT_TEMPLATE_PATH.is_file():
        print(f"ERROR: Prompt template not found at: {PROMPT_TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    # (load_prompt_template will exit if "{entity_type}" is missing.)

    # 3) Ensure output directory exists.
    OUTPUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) Open the output .txt for writing.
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as outfile:
        # 5) Iterate over each entity type, call the LLM, parse JSON, and write output.
        for etype in tqdm(entities, desc="Defining entity types"):
            # 5.a) Substitute the placeholder.
            prompt_text = prompt_template.replace("{entity_type}", etype)

            definition = ""
            # 5.b–e) Retry loop: up to 10 attempts to parse valid JSON:
            for attempt in range(1, 11):
                # 5.b) Call Deepseek (via Ollama).
                _, raw_output = call_deepseek(prompt_text)

                # 5.c) Clean out any <think>…</think> tags / extra whitespace.
                cleaned = clean_output(raw_output) if raw_output else ""

                # 5.d) Extract only the JSON object (“{ … }”) from the LLM’s reply.
                json_snippet = extract_json_object(cleaned)

                # 5.e) Parse the JSON and get the "definition" field.
                if json_snippet:
                    try:
                        parsed = json.loads(json_snippet)
                        if isinstance(parsed, dict) and "definition" in parsed:
                            definition = parsed["definition"].strip()
                            break  # success!
                    except json.JSONDecodeError:
                        # Failed to parse; will retry if attempts remain
                        pass

                # If this was the last attempt, we give up and leave definition = ""
                if attempt == 10:
                    break

            # 5.f) Write “entity_type: definition” (even if definition is empty).
            outfile.write(f"{etype}: {definition}\n")

    print(f"✅ Finished. Definitions saved to: {OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    main()
