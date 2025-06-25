# new_27 ke sabhi folders ke name waali document files ko preprocess krna and analysis report banana.
import os
import csv
from collections import OrderedDict
import sys
import re
from tqdm import tqdm
# Import preprocessing pipeline and processor
from preprocessing import process_document, ImproveParaProcessor

# User-configurable paths
# Set these paths before running the script:
FOLDERS_DIR = "output/new_27"
TEXTS_DIR = "datasets/judgements"
PREPROC_DIR = "output/preprocessed_judgments"
OUTPUT_CSV = "output/preprocessed_judgments"
PREPROCESSING_DIR = "scripts/filox"


# Ensure the preprocessing directory is on Python path
sys.path.insert(0, PREPROCESSING_DIR)



def describe_file(text):
    """
    Compute descriptive statistics for preprocessed text, where each non-empty line is treated as a sentence,
    but lines that are purely a number followed by a period (e.g., "1.") or that contain no word tokens are skipped.
    Returns a dict of metrics.
    """
    sentences = []
    word_counts = []
    for line in text.splitlines():
        stripped = line.strip()
        # skip empty lines and lines like "<number>."
        if not stripped or re.match(r'^\d+\.$', stripped):
            continue
        # find word tokens in the line
        tokens = re.findall(r"\b\w+\b", stripped)
        # skip lines with zero words
        if not tokens:
            continue
        sentences.append(stripped)
        word_counts.append(len(tokens))

    # total words across all sentences
    num_sentences = len(sentences)
    num_words = sum(word_counts)
    avg_word_length = sum(len(w) for w in re.findall(r"\b\w+\b", text)) / num_words if num_words else 0

    if word_counts:
        avg_sent_len = sum(word_counts) / num_sentences
        min_sent_len = min(word_counts)
        max_sent_len = max(word_counts)
    else:
        avg_sent_len = min_sent_len = max_sent_len = 0

    # vocabulary from entire text (excluding skipped markers)
    words = re.findall(r"\b\w+\b", text)
    vocab = set(w.lower() for w in words)
    vocab_size = len(vocab)
    lexical_diversity = vocab_size / num_words if num_words else 0

    return {
        'num_words': num_words,
        'num_sentences': num_sentences,
        'avg_sentence_length': avg_sent_len,
        'min_sentence_length': min_sent_len,
        'max_sentence_length': max_sent_len,
        'avg_word_length': avg_word_length,
        'vocab_size': vocab_size,
        'lexical_diversity': lexical_diversity
    }


def get_folder_names(dir_path):
    """
    Return a list of immediate folder names in the given directory.
    """
    return [name for name in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, name))]


def main():
    # Ensure preprocessed directory exists
    os.makedirs(PREPROC_DIR, exist_ok=True)

    # Handle OUTPUT_CSV path: if it's a directory, place default CSV inside
    if os.path.isdir(OUTPUT_CSV):
        default_name = "dataset_description.csv"
        dest_dir = OUTPUT_CSV
        OUTPUT_CSV_path = os.path.join(dest_dir, default_name)
    else:
        OUTPUT_CSV_path = OUTPUT_CSV
        parent = os.path.dirname(OUTPUT_CSV_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # Initialize processor with only contractions & abbreviations
    processor = ImproveParaProcessor()
    for key in processor.options:
        processor.options[key] = False
    processor.options['expand_contractions'] = True
    processor.options['remove_abbreviations'] = True

    folder_names = get_folder_names(FOLDERS_DIR)
    rows = []

    # Process with progress bar
    for name in tqdm(folder_names, desc="Processing files", unit="file"):
        orig_file = os.path.join(TEXTS_DIR, f"{name}.txt")
        preproc_file = process_document(orig_file, PREPROC_DIR, processor)

        with open(preproc_file, 'r', encoding='utf-8') as f:
            text = f.read()

        metrics = describe_file(text)
        row = {'folder': name, 'file': os.path.basename(preproc_file)}
        row.update(metrics)
        rows.append(row)

    # Write CSV with progress bar
    with open(OUTPUT_CSV_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'folder', 'file', 'num_words', 'num_sentences',
            'avg_sentence_length', 'min_sentence_length', 'max_sentence_length',
            'avg_word_length', 'vocab_size', 'lexical_diversity'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in tqdm(rows, desc="Writing CSV", unit="row"):
            writer.writerow(r)

    print(f"Preprocessed files (only contractions & abbreviations) saved in: {PREPROC_DIR}")
    print(f"Analysis CSV saved as: {OUTPUT_CSV_path}")

if __name__ == '__main__':
    main()
