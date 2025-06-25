#!/usr/bin/env python3
"""
Generate one-line natural language definitions for relation labels using LLM assistance.
"""

import csv
import json
import sys
from pathlib import Path
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / 'entity_types'))

from entity_relation_extraction import call_deepseek, load_prompt_template

# CONFIGURATION
INPUT_CSV = Path("output/relations/final_relations.csv")  # Uses canonical names
PROMPT_PATH = Path("prompts/relations_definition.txt")
OUTPUT_CSV = Path("output/relations/definitions.csv")
OUTPUT_TXT = Path("datasets/relations.txt")
MAX_LLM_RETRIES = 3


def generate_definition_prompt(template: str, relation: str) -> str:
    return template.format(relation=relation)


def main():
    # Validate files
    if not PROMPT_PATH.exists():
        print(f"❌ Prompt file missing: {PROMPT_PATH}")
        sys.exit(1)

    prompt_template = load_prompt_template(str(PROMPT_PATH))
    if '{relation}' not in prompt_template:
        print("❌ Prompt template must include `{relation}`")
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"❌ Input CSV missing: {INPUT_CSV}")
        sys.exit(1)

    # Load relations from input CSV
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        relations = [row['Relations'] for row in reader if row.get('Relations')]

    definitions = []

    for relation in tqdm(relations, desc="Generating definitions"):
        prompt = generate_definition_prompt(prompt_template, relation)
        print(f"\n--- Prompt for relation: {relation} ---")
        print(prompt)

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                # Call the LLM
                output = call_deepseek(prompt)

                # Unpack text from tuple or list, if necessary
                if isinstance(output, (list, tuple)):
                    response_text = output[1]
                else:
                    response_text = str(output)

                print(f"Response (raw): {response_text}")

                # Extract JSON substring
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = response_text[start:end]
                else:
                    json_str = response_text.strip()

                # Parse JSON
                obj = json.loads(json_str)

                # Append to definitions
                definitions.append((obj["relation"], obj["definition"]))
                print("definitions-->", definitions)
                break

            except Exception as e:
                print(f"Attempt {attempt} failed for relation '{relation}': {e}")
                # On final failure, record empty definition
                if attempt == MAX_LLM_RETRIES:
                    print(f"⚠️ Failed to generate definition for: {relation}")
                    definitions.append((relation, ""))
                    print("definitions-->", definitions)
                    break
                # otherwise, retry

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["relation", "definition"])
        for relation, definition in definitions:
            writer.writerow([relation, definition])

    # Save to TXT
    with open(OUTPUT_TXT, "w", encoding='utf-8') as f_txt:
        for relation, definition in definitions:
            f_txt.write(f"{relation}: {definition}\n")

    print(f"✅ Definitions written to:\n  • {OUTPUT_CSV.name}\n  • {OUTPUT_TXT.name}")


if __name__ == "__main__":
    main()
