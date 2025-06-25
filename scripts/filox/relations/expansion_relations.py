#!/usr/bin/env python3
import sys
import json, csv, time
from pathlib import Path
from tqdm import tqdm
import re

# ─── Make sure we can import any helpers from the sibling folder ───
THIS_DIR = Path(__file__).parent
ENTITY_TYPES_DIR = THIS_DIR.parent / "entity_types"
sys.path.insert(0, str(ENTITY_TYPES_DIR))

from entity_relation_extraction import (
    load_document,
    clean_output,
    call_deepseek,
    load_entity_types,
    chunk_entity_types,
    load_prompt_template
)

# ──────── HARD-CODED PATHS ───────────────────────────────────────────
PREPROCESSED_ROOT = Path("output/new_27")
OUTPUT_DIR        = Path("output/relations")
ENTITY_FILE       = Path("datasets/entity_types_100.txt")  # Type: Definition list
PROMPT_TEMPLATE   = Path("prompts/prompt_relation.txt")    # with {entity_descriptions}, {sentence}, {context_text}
MAX_RETRIES       = 10
# ────────────────────────────────────────────────────────────────────

def build_paragraph_texts(sentences):
    paras = {}
    for _, pnum, sent in sentences:
        paras.setdefault(pnum, []).append(sent)
    return {p: " ".join(lines) for p, lines in paras.items()}


def extract_all_json_arrays(text):
    """
    Scan the text and extract all valid JSON arrays.
    """
    decoder = json.JSONDecoder()
    idx = 0
    arrays = []
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text[idx:])
            if isinstance(obj, list):
                arrays.append(obj)
                idx += end
            else:
                idx += 1
        except json.JSONDecodeError:
            idx += 1
    return arrays


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1) Load entity-type definitions and the prompt
    entity_types        = load_entity_types(ENTITY_FILE)
    entity_descriptions = "\n".join(f"- {k}: {v}" for k, v in entity_types.items())
    prompt_template     = load_prompt_template(PROMPT_TEMPLATE)

    # 2) For each folder…
    for folder in tqdm(sorted(PREPROCESSED_ROOT.iterdir()), desc="Judgements"):
        if not folder.is_dir():
            continue

        fname = f"preprocessed_{folder.name}.txt"
        doc   = folder / fname
        if not doc.exists():
            tqdm.write(f"Skipping {folder.name}: no {fname}")
            continue

        # 3) parse sentences & build contexts
        sentences = load_document(doc)  # [(sent_no, para_no, sentence), …]
        para_text = build_paragraph_texts(sentences)

        all_relations = []
        for sent_no, para_no, sent in tqdm(sentences, desc=f"  Extract {folder.name}"):
            ctx = para_text.get(para_no, "")
            prompt = prompt_template.format(
                entity_descriptions=entity_descriptions.replace('"', r'\"'),
                sentence=sent.replace('"', r'\"'),
                context_text=ctx.replace('"', r'\"')
            )

            # 4) LLM + retry
            for i in range(1, MAX_RETRIES + 1):
                raw = call_deepseek(prompt)

                # Normalize raw to string containing the LLM text
                if isinstance(raw, (tuple, list)) and len(raw) > 0:
                    if isinstance(raw[0], str):
                        raw_text = raw[0]
                    else:   
                        raw_text = json.dumps(raw[0])
                elif hasattr(raw, 'text'):
                    raw_text = raw.text
                elif isinstance(raw, dict) and 'choices' in raw:
                    raw_text = raw['choices'][0].get('text', '')
                else:
                    raw_text = str(raw)

                # print(f"\n🟡 Prompt sent to LLM (sentence #{sent_no}, para #{para_no}):\n{prompt[:500]}")
                # print("--- Raw LLM Output (first 500 chars) ---")
                # print(raw_text[:500])

                json_arrays = extract_all_json_arrays(raw_text)
                # print("--- Extracted JSON Candidates ---")
                # for j in json_arrays:
                #     print(json.dumps(j, indent=2))
                # print("-------------------------\n")

                if json_arrays:
                    for rels in json_arrays:
                        if isinstance(rels, list):
                            for r in rels:
                                if isinstance(r, dict):
                                    # print("✅ Relation Found:", r.get("head_type"), "->", r.get("relation"), "->", r.get("tail_type"))
                                    r.update({
                                        "sentence":         sent,
                                        "sentence_number":  str(sent_no),
                                        "paragraph_number": str(para_no)
                                    })
                            all_relations.extend(rels)
                    break  # break after successful parse of all arrays

                else:
                    print(f"❌ JSON decode failed on try {i}/{MAX_RETRIES}")
                    if i < MAX_RETRIES:
                        time.sleep(1)
                        continue

        # 5) dump JSON + CSV
        outdir = OUTPUT_DIR / folder.name / "results"
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{folder.name}.json", "w") as jf:
            json.dump(all_relations, jf, indent=2)
        with open(outdir / f"{folder.name}.csv", "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "head_type", "relation", "tail_type",
                    "sentence", "sentence_number", "paragraph_number"
                ]
            )
            writer.writeheader()
            for r in all_relations:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print("Done. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
