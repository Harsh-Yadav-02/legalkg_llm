#!/usr/bin/env python3
import sys
import json
import csv
import subprocess
import re
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration: Hard-coded paths & settings ===
PREPROCESSED_ROOT = Path("output/window_comparison/comp_10")
OUTPUT_ROOT       = Path("output/window_comparison/entities_and_relations_chunk_2")
ENTITY_FILE       = Path("datasets/entity_types_100.txt")
FEW_SHOT_FILE     = Path("datasets/few_shot_examples_triple.txt")
PROMPT_TEMPLATES = {
    'sentence': Path("prompts/prompt_sentence.txt"),
    'paragraph': Path("prompts/prompt_paragraph.txt"),
    'chunk': Path("prompts/prompt_chunk.txt")
}

MAX_RETRIES      = 10
NUM_WORKERS      = 640  # Now used for ThreadPool
CHUNK_SIZE       = 100
EXTRACTION_LEVEL = 'chunk'
LOG_PROMPT       = True
LOG_DIR          = Path("./sent_prompt_and_response")

# === Import helpers ===
THIS_DIR = Path(__file__).resolve().parent
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

# === Process a single sentence ===
def process_sentence(args):
    sent_no, para_no, sent, entity_types, few_shot_examples, prompt_template, paragraph_texts = args
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
        sentence=sent,
        sentence_number=sent_no,
        context_text=ctx
    )

    results = []
    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_deepseek(prompt)
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

# === Process a single document in parallel ===
def process_file(doc_path, entity_types, few_shot_examples, prompt_template, paragraph_texts):
    sentences = load_document(doc_path)
    results = []
    # Prepare args for pool
    task_args = [
        (sent_no, para_no, sent, entity_types, few_shot_examples, prompt_template, paragraph_texts)
        for sent_no, para_no, sent in sentences
    ]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_sentence = {executor.submit(process_sentence, arg): arg for arg in task_args}
        for future in tqdm(as_completed(future_to_sentence), total=len(task_args), desc=f"  Sentences in {doc_path.name}"):
            try:
                sentence_results = future.result()
                results.extend(sentence_results)
            except Exception as e:
                print(f"⚠️ Error processing sentence: {e}")
    return results

# === Main entrypoint ===
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entity_types = load_entity_types(ENTITY_FILE)
    few_shot_examples = load_few_shot_examples(FEW_SHOT_FILE)
    global entity_descriptions
    entity_descriptions = "\n".join(f"- {k}: {v}" for k, v in entity_types.items())
    prompt_template = load_prompt_template(PROMPT_TEMPLATES[EXTRACTION_LEVEL])

    for folder in tqdm(sorted(PREPROCESSED_ROOT.iterdir()), desc="Documents"):
        if not folder.is_dir():
            continue
        doc_file = folder / f"preprocessed_{folder.name}.txt"
        if not doc_file.exists():
            tqdm.write(f"Skipping {folder.name}: no preprocessed file")
            continue

        sentences = load_document(doc_file)
        paragraph_texts = {}
        for _, p, s in sentences:
            paragraph_texts.setdefault(p, []).append(s)
        paragraph_texts = {p: " ".join(lines) for p, lines in paragraph_texts.items()}

        relations = process_file(doc_file, entity_types, few_shot_examples, prompt_template, paragraph_texts)

        out_dir = OUTPUT_ROOT / folder.name / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{folder.name}.json", 'w', encoding='utf-8') as jf:
            json.dump(relations, jf, indent=2)
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
