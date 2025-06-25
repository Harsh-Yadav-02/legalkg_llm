#!/usr/bin/env python3
"""
Wrapper script to preprocess judgements and extract entities using Deepseek locally.

For each text file in the input directory:
 1. Invoke the preprocessing script to produce a sentence-wise, cleaned file.
 2. Call the entity-extraction script on the preprocessed file to extract entities, types, and abbreviations.
 3. Save the resulting CSV and JSON outputs under a per-judgement results folder.
 4. Store sent prompts and LLM logs within each judgement's output directory.
 5. Display progress via tqdm.

Usage:
  python entity_extraction_pipeline.py --input_dir /path/to/judgements --output_dir /path/to/output --preprocessing_script /path/to/preprocessing.py --entity_script /path/to/entity_relation_extraction.py --few_shot_file /path/to/few_shot.txt --entity_file /path/to/entity_types.txt --prompt_template /path/to/prompt_template.txt --num_workers 16 --chunk_size 10 --extraction_level paragraph --log_prompt [--contractions --abbrev]
"""
import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import json, csv


def run_preprocessing(preproc_script, input_file, output_dir, flags):
    """
    Runs the preprocessing script on a single judgement file.
    Returns the path to the preprocessed output.
    """
    cmd = [sys.executable, str(preproc_script), str(input_file), str(output_dir)] + flags
    subprocess.run(cmd, check=True)
    return output_dir / f"preprocessed_{input_file.name}"


def run_extraction(entity_script, document_file, entity_file,
                   prompt_template, out_json, out_csv, num_workers,
                   chunk_size, extraction_level, log_flag, work_dir):
    """
    Runs the entity relation extraction script on a preprocessed document.
    Outputs JSON and CSV into specified paths.
    Logs prompts and raw LLM responses under work_dir.
    """
    template_flag = '--prompt_template_file_chunk' if extraction_level == 'chunk' else '--prompt_template_file_paragraph'
    cmd = [
        sys.executable, str(entity_script),
        '--document_file', str(document_file),
        # '--few_shot_file', str(few_shot_file),
        '--entity_file', str(entity_file),
        '--extraction_level', extraction_level,
        template_flag, str(prompt_template),
        '--output_json_file', str(out_json),
        '--output_csv_file', str(out_csv),
        '--num_workers', str(num_workers),
        '--chunk_size', str(chunk_size)
    ]
    if log_flag:
        cmd.append('--log_prompt')
    # subprocess.run(cmd, check=True, cwd=str(work_dir))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: preprocess judgements and extract entities via Deepseek."
    )
    parser.add_argument('--input_dir', required=True, type=Path,
                        help="Directory containing judgement text files.")
    parser.add_argument('--output_dir', required=True, type=Path,
                        help="Directory to write outputs (one subfolder per judgement).")
    parser.add_argument('--preprocessing_script', required=True, type=Path,
                        help="Path to the preprocessing.py script.")
    parser.add_argument('--entity_script', required=True, type=Path,
                        help="Path to the entity_relation_extraction.py script.")
    # parser.add_argument('--few_shot_file', required=True, type=Path,
                        # help="Path to few-shot examples text file.")
    parser.add_argument('--entity_file', required=True, type=Path,
                        help="Path to the entity types definitions file.")
    parser.add_argument('--prompt_template', required=True, type=Path,
                        help="Path to the prompt template for extraction.")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="Number of parallel workers for extraction.")
    parser.add_argument('--chunk_size', default=100, type=int,
                        help="Number of entity types per chunk in prompts.")
    parser.add_argument('--extraction_level', choices=['sentence','paragraph','chunk'],
                        default='paragraph', help="Context level for LLM prompts.")
    parser.add_argument('--log_prompt', action='store_true',
                        help="Enable logging of sent prompts and LLM responses.")

    # Parse known args; treat any extra flags as preprocessing flags
    args, extra_flags = parser.parse_known_args()
    args.preproc_flags = [f for f in extra_flags if f != '--preproc_flags']

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list(args.input_dir.glob('*.txt'))
    for judgement in tqdm(files, desc="Processing judgements"):
        work_dir = args.output_dir / judgement.stem
        work_dir.mkdir(exist_ok=True)
        preprocessed = run_preprocessing(
            args.preprocessing_script, judgement, work_dir, args.preproc_flags
        )

        results_dir = work_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        out_json = results_dir / f"{judgement.stem}.json"
        out_csv = results_dir / f"{judgement.stem}.csv"

        run_extraction(
            args.entity_script,
            preprocessed,
            # args.few_shot_file,
            args.entity_file,
            args.prompt_template,
            out_json,
            out_csv,
            args.num_workers,
            args.chunk_size,
            args.extraction_level,
            args.log_prompt,
            work_dir
        )
        with open(out_json, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
        # overwrite CSV with only the selected columns
        with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
            fieldnames = ['entity', 'entity_type', 'sentence', 'context_text', 'sentence_number', 'paragraph_number']
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for obj in data:
                # serialize abbreviations list as comma-separated string
                abbr = obj.get('abbreviations', [])
                if isinstance(abbr, list):
                    abbr = ", ".join(abbr)
                writer.writerow({
                    'entity':            obj.get('entity', ''),
                    'entity_type':       obj.get('entity_type', ''),
                    'sentence':        obj.get('sentence', ''),
                    'context_text':        obj.get('context_text', ''),
                    'sentence_number': obj.get('sentence_number', ''),
                    'paragraph_number':  obj.get('paragraph_number', '')
                })    

    print("All judgements processed. Results are in:", args.output_dir)

if __name__ == '__main__':
    main()
