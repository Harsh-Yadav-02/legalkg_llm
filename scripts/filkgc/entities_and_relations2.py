#!/usr/bin/env python3
import sys
import json
import csv
import subprocess
import re
import time
from pathlib import Path
from tqdm import tqdm

# === Configuration: Hard-coded paths & settings ===
PREPROCESSED_ROOT = Path("output/try")       # Root dir of preprocessed files
OUTPUT_ROOT       = Path("output/entities_and_relations2")    # Base output dir
ENTITY_FILE       = Path("datasets/entity_types_100.txt")  # Type: Definition list
FEW_SHOT_FILE     = Path("datasets/few_shot_examples_triple.txt")
RELATION_FILE     = Path("datasets/relations.txt")   # New: Relation: Definition list
PROMPT_TEMPLATES = {
    'sentence': Path("prompts/prompt_sentence.txt"),
    'paragraph': Path("prompts/prompt_paragraph_with_relations.txt"),
    'chunk': Path("prompts/prompt_chunk.txt")
}

MAX_RETRIES      = 10
NUM_WORKERS      = 32  # Placeholder; multiprocessing not implemented in this version
CHUNK_SIZE       = 100  # entity types per chunk
EXTRACTION_LEVEL = 'paragraph'  # 'sentence' | 'paragraph' | 'chunk'
LOG_PROMPT       = True
LOG_DIR          = Path("./sent_prompt_and_response")

# === Import helpers from original script (adjusted path) ===
THIS_DIR = Path(__file__).resolve().parent
# Add filox/entity_types folder to module search path
sys.path.insert(0, str(THIS_DIR.parent / 'filox' / 'entity_types'))
from entity_relation_extraction import (
    load_document,
    load_few_shot_examples,
    clean_output,
    call_deepseek,
    load_entity_types,
    chunk_entity_types,
    load_prompt_template
)

# === New loader for relations definitions ===
def load_relation_definitions(path: Path) -> dict:
    """
    Load relations definitions from a text file with 'relation: definition' per line.
    Returns a dict mapping relation -> definition.
    """
    rel_defs = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                rel, definition = line.strip().split(':', 1)
                rel_defs[rel.strip()] = definition.strip()
    return rel_defs

# === Utility to extract multiple JSON arrays robustly ===
def extract_all_json_arrays(text):
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

# === Process a single document sequentially ===
def process_file(doc_path, entity_types, relation_defs, few_shot_examples, prompt_template, paragraph_texts):
    sentences = load_document(doc_path)
    results = []
    for sent_no, para_no, sent in tqdm(sentences, desc=f"  Sentences in {doc_path.name}"):
        # build context
        if EXTRACTION_LEVEL == 'sentence':
            ctx = sent
        elif EXTRACTION_LEVEL == 'paragraph':
            ctx = paragraph_texts.get(para_no, sent)
        else:  # 'chunk'
            parts = []
            for p in range(para_no - 2, para_no + 3):
                if p in paragraph_texts:
                    parts.append(paragraph_texts[p])
            ctx = " ".join(parts) if parts else sent

        # build prompt
        prompt = prompt_template.format(
            few_shot_examples=few_shot_examples,
            entity_descriptions=entity_descriptions,
            relation_definitions=relation_descriptions,
            sentence=sent,
            sentence_number=sent_no,
            context_text=ctx
        )
        # print(prompt)

        raw_text = None
        # LLM call with retries
        for attempt in range(1, MAX_RETRIES + 1):
            raw = call_deepseek(prompt)
            print(raw)
            # normalize output
            if isinstance(raw, (tuple, list)) and raw:
                raw_text = raw[0] if isinstance(raw[0], str) else json.dumps(raw[0])
            elif hasattr(raw, 'text'):
                raw_text = raw.text
            elif isinstance(raw, dict) and 'choices' in raw:
                raw_text = raw['choices'][0].get('text', '')
            else:
                raw_text = str(raw)

            arrays = extract_all_json_arrays(raw_text)
            if arrays:
                for arr in arrays:
                    for r in arr:
                        if isinstance(r, dict):
                            # enrich relation dict with metadata
                            r.update({
                                'head': r.get('head', ''),
                                'head_type': r.get('head_type', ''),
                                'relation': r.get('relation', ''),
                                'tail': r.get('tail', ''),
                                'tail_type': r.get('tail_type', ''),
                                'relation_definition': r.get('relation_definition', ''),
                                'sentence': sent,
                                'sentence_number': sent_no,
                                'paragraph_number': para_no,
                                'context_text': ctx
                            })
                            results.append(r)
                break
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"❌ Failed JSON parse after {MAX_RETRIES} attempts for sentence {sent_no}")
    return results

# === Main entrypoint ===
def main():
    # prepare directories
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # load configs
    entity_types = load_entity_types(ENTITY_FILE)
    few_shot_examples = load_few_shot_examples(FEW_SHOT_FILE)
    relation_defs = load_relation_definitions(RELATION_FILE)

    global entity_descriptions, relation_descriptions
    entity_descriptions = "\n".join(f"- {k}: {v}" for k, v in entity_types.items())
    relation_descriptions = "\n".join(f"- {r}: {d}" for r, d in relation_defs.items())

    prompt_template = load_prompt_template(PROMPT_TEMPLATES[EXTRACTION_LEVEL])

    # iterate over each judgment folder
    for folder in tqdm(sorted(PREPROCESSED_ROOT.iterdir()), desc="Documents"):
        if not folder.is_dir():
            continue
        doc_file = folder / f"preprocessed_{folder.name}.txt"
        if not doc_file.exists():
            tqdm.write(f"Skipping {folder.name}: no preprocessed file")
            continue

        # build paragraph texts
        sentences = load_document(doc_file)
        paragraph_texts = {}
        for _, p, s in sentences:
            paragraph_texts.setdefault(p, []).append(s)
        paragraph_texts = {p: " ".join(lines) for p, lines in paragraph_texts.items()}

        # process document
        relations = process_file(doc_file, entity_types, relation_defs, few_shot_examples, prompt_template, paragraph_texts)

        # write outputs
        out_dir = OUTPUT_ROOT / folder.name / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        # JSON
        with open(out_dir / f"{folder.name}.json", 'w', encoding='utf-8') as jf:
            json.dump(relations, jf, indent=2)
        # CSV
        fieldnames = [
            'head','head_type','relation','tail','tail_type','relation_definition',
            'sentence','sentence_number','paragraph_number'
        ]
        with open(out_dir / f"{folder.name}.csv", 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in relations:
                writer.writerow({k: r.get(k, '') for k in fieldnames})

    print("Done. Results in", OUTPUT_ROOT)

if __name__ == '__main__':
    main()
