import argparse
import csv
import json
import subprocess
import multiprocessing
import re
import os
from tqdm import tqdm

def load_entity_types(entity_file):
    """Load valid entity types and their definitions from a file.
    
    Each line in the file should be formatted as:
      EntityType: Definition text...
    """
    entity_dict = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(":", 1)  # Split at the first colon
                if len(parts) == 2:
                    entity, definition = parts
                    entity_dict[entity.strip()] = definition.strip()
                else:
                    entity_dict[line] = "No definition provided"
    return entity_dict

def load_few_shot_examples(examples_file):
    """Load few-shot examples from a file as a text block."""
    try:
        with open(examples_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print("Error reading few-shot file:", e)
        return ""

def load_prompt_template(template_file):
    """Load the prompt template from a file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print("Error reading prompt template file:", e)
        return None

def chunk_entity_types(entity_types, chunk_size=10):
    """Yield smaller dictionaries of entity types of size chunk_size."""
    items = list(entity_types.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i+chunk_size])

def load_document(document_file):
    """
    Load document sentences from a file (one sentence per line) and
    return a list of tuples: (sentence_number, paragraph_number, sentence).
    
    The paragraph_number starts at 0. When a line contains only a number followed by a period (e.g., "1."),
    that line is treated as a paragraph indicator and subsequent sentences are assigned that paragraph number.
    """
    sentences = []
    current_paragraph = 0
    with open(document_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Check if the line is a paragraph marker (e.g., "1.", "2.", etc.)
            if re.fullmatch(r'\d+\.', line):
                try:
                    current_paragraph = int(line[:-1])
                except ValueError:
                    current_paragraph = 0
                continue    
                # Include the marker as a sentence (optional)
                # sentences.append((idx + 1, current_paragraph, line))
            else:
                sentences.append((idx + 1, current_paragraph, line))
    return sentences

def clean_output(output):
    """
    Remove any <think>...</think> tags and extra whitespace from the output.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    return cleaned.strip()

def extract_json(text):
    """
    Extract a JSON array from the text using a regular expression.
    If found, returns the JSON substring; otherwise, returns the original text.
    """
    json_objects = []
    matches = re.findall(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                json_objects.extend(parsed)
        except json.JSONDecodeError:
            continue
    return json_objects
    # match = re.search(r'(\[.*\])', text, re.DOTALL)
    # if match:
    #     return match.group(1)
    # return text

def construct_prompt(few_shot_examples, entity_types, sentence, sentence_number, context_text=None, prompt_template=None):

    # Prepare entity descriptions formatted as a bullet list.
    entity_descriptions = "\n".join([f"- {etype}: {definition}" for etype, definition in entity_types.items()])
    
    if prompt_template:
        prompt = prompt_template.format(
            few_shot_examples=few_shot_examples,
            entity_descriptions=entity_descriptions,
            sentence_number=sentence_number,
            sentence=sentence,
            context_text=context_text if context_text else ""
        )
        return prompt
    else:
        print("Invalid PROMPT path")

def call_deepseek(prompt):
    """
    Call Deepseek via Ollama by invoking a subprocess.
    Returns a tuple: (json_text, raw_output)
    where json_text is the cleaned JSON extraction and raw_output is the raw response (with <think> tags).
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:14b'],  # Updated model name
            input=prompt,
            text=True,
            capture_output=True,
            timeout=None,
            encoding='utf-8'
        )
        if result.returncode == 0:
            raw_output = result.stdout.strip()
            cleaned = clean_output(raw_output)
            json_text = extract_json(cleaned)
            return json_text, raw_output
        else:
            print("Error calling Deepseek:", result.stderr)
            return None, None
    except Exception as e:
        print("Exception while calling Deepseek:", str(e))
        return None, None

def worker(input_queue, output_queue, few_shot_examples, entity_types, chunk_size, prompt_template,
           log_prompt, log_lock, extraction_level, paragraph_texts):
    while True:
        item = input_queue.get()
        if item is None:
            break  # Sentinel received; exit.
        sentence_number, paragraph_number, sentence = item

        # Compute context based on extraction level.
        if extraction_level == "sentence":
            context_text = sentence
        elif extraction_level == "paragraph":
            context_text = paragraph_texts.get(paragraph_number, sentence)
        elif extraction_level == "chunk":
            chunk_text_parts = []
            # Include paragraphs from (current - 3) to (current + 3)
            for p in range(paragraph_number - 2, paragraph_number + 3):
                if p in paragraph_texts:
                    chunk_text_parts.append(paragraph_texts[p])
            context_text = " ".join(chunk_text_parts) if chunk_text_parts else sentence
        else:
            context_text = sentence

        all_results = []
        # Process the sentence in chunks of entity types.
        for entity_chunk in chunk_entity_types(entity_types, chunk_size):
            # Build a prompt using this chunk.
            prompt = construct_prompt(
                few_shot_examples, entity_chunk, sentence, sentence_number,
                context_text=context_text,
                prompt_template=prompt_template
            )
            # Log the prompt if logging is enabled.
            if log_prompt and log_lock is not None:
                with log_lock:
                    with open("./sent_prompt_and_response/sent_prompt.txt", "a", encoding="utf-8") as f:
                        f.write(f"DEBUG: Final prompt for sentence {sentence_number} (Paragraph {paragraph_number}):\n")
                        f.write(prompt + "\n")
                        f.write("=" * 60 + "\n")
            # Call the LLM.
            result, raw_output = call_deepseek(prompt)
            # Log the raw LLM response in a separate log file.
            if log_prompt and log_lock is not None and raw_output is not None:
                with log_lock:
                    with open("./sent_prompt_and_response/llm_logs.txt", "a", encoding="utf-8") as log_file:
                        log_file.write(f"DEBUG: LLM response for sentence {sentence_number} (Paragraph {paragraph_number}):\n")
                        log_file.write(raw_output + "\n")
                        log_file.write("=" * 60 + "\n")
            if result:
                result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL)
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        all_results.extend(parsed)
                except json.JSONDecodeError:
                    print("JSON decode error for sentence:", sentence)
                    print("Raw fetched output:", result)
        # Optionally, deduplicate results.
        unique_results = {json.dumps(obj, sort_keys=True): obj for obj in all_results}.values()
        for obj in unique_results:
            if not isinstance(obj, dict):
                continue
            obj.setdefault('sentence', sentence)
            obj.setdefault('sentence_number', sentence_number)
            obj.setdefault('paragraph_number', paragraph_number)
            obj['context_text'] = context_text
            output_queue.put(obj)
    # Signal worker completion.
    output_queue.put("WORKER_DONE")

def process_document(document_file, few_shot_file, entity_file, prompt_template_files,
                     output_json_file, output_csv_file, num_workers, log_prompt, chunk_size, extraction_level):
    """
    Orchestrates:
      - Loading input files.
      - Preparing context based on extraction level.
      - Starting persistent worker processes.
      - Distributing (sentence_number, paragraph_number, sentence) tuples for processing.
      - Collecting extracted relations.
      - Writing them to JSON and CSV.
    """
    entity_types = load_entity_types(entity_file)
    few_shot_examples = load_few_shot_examples(few_shot_file)

    # Select the appropriate prompt template based on extraction_level.
    prompt_template = None
    if extraction_level == "sentence":
        prompt_template = load_prompt_template(prompt_template_files.get('sentence', ""))
    elif extraction_level == "paragraph":
        prompt_template = load_prompt_template(prompt_template_files.get('paragraph', ""))
    elif extraction_level == "chunk":
        prompt_template = load_prompt_template(prompt_template_files.get('chunk', ""))
    
    sentence_tuples = load_document(document_file)
    
    # Create a mapping from paragraph number to full paragraph text.
    paragraphs = {}
    for _, para_num, sent in sentence_tuples:
        paragraphs.setdefault(para_num, []).append(sent)
    # Join sentences in each paragraph to form full paragraphs.
    paragraph_texts = {p: " ".join(sent_list) for p, sent_list in paragraphs.items()}

    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()
    log_lock = multiprocessing.Lock()

    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker,
                                    args=(input_queue, output_queue, few_shot_examples, entity_types,
                                          chunk_size, prompt_template, log_prompt, log_lock,
                                          extraction_level, paragraph_texts))
        p.start()
        workers.append(p)

    for item in sentence_tuples:
        input_queue.put(item)
    
    # Send sentinel values.
    for _ in range(num_workers):
        input_queue.put(None)

    results = []
    finished_workers = 0
    while finished_workers < num_workers:
        item = output_queue.get()
        if item == "WORKER_DONE":
            finished_workers += 1
        elif item is not None:
            results.append(item)

    for p in workers:
        p.join()

    # Create output directories if needed.
    json_dir = os.path.dirname(output_json_file)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)

    csv_dir = os.path.dirname(output_csv_file)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    with open(output_json_file, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=4)

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['head', 'head_type', 'relation', 'tail', 'tail_type', 
                      'relation_definition', 'sentence', 'sentence_number', 'paragraph_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'head': item.get('head', ''),
                'head_type': item.get('head_type', ''),
                'relation': item.get('relation', ''),
                'tail': item.get('tail', ''),
                'tail_type': item.get('tail_type', ''),
                'relation_definition': item.get('relation_definition', ''),
                'sentence': item.get('sentence', ''),
                'sentence_number': item.get('sentence_number', ''),
                'paragraph_number': item.get('paragraph_number', '')
            })

def main():
    parser = argparse.ArgumentParser(
        description="Extract legal relations using Deepseek via Ollama with support for different context levels."
    )
    parser.add_argument('--log_prompt', action='store_true', help="Save the prompt and LLM response in log files.")
    parser.add_argument('--document_file', type=str, required=True,
                        help="Path to the input document file (one sentence per line).")
    parser.add_argument('--few_shot_file', type=str, required=False,
                        help="Path to the few-shot examples file.")
    parser.add_argument('--entity_file', type=str, required=True,
                        help="Path to the file containing entity types and definitions (one per line, formatted as 'EntityType: Definition').")
    # New arguments for extraction level and separate prompt templates.
    parser.add_argument('--extraction_level', choices=['sentence', 'paragraph', 'chunk'], default='sentence',
                        help="Level of context to include in the prompt: sentence, paragraph, or chunk (3 paragraphs above and below).")
    parser.add_argument('--prompt_template_file_sentence', type=str, default="",
                        help="Path to the prompt template file for sentence-level extraction.")
    parser.add_argument('--prompt_template_file_paragraph', type=str, default="",
                        help="Path to the prompt template file for paragraph-level extraction.")
    parser.add_argument('--prompt_template_file_chunk', type=str, default="",
                        help="Path to the prompt template file for chunk-level extraction.")
    parser.add_argument('--output_json_file', type=str, default='output.json',
                        help="Path for the output JSON file.")
    parser.add_argument('--output_csv_file', type=str, default='output.csv',
                        help="Path for the output CSV file.")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of persistent worker processes.")
    parser.add_argument('--chunk_size', type=int, default=10,
                        help="Number of entity types per chunk when constructing the prompt.")
    args = parser.parse_args()

    # Map prompt template file paths to extraction level keys.
    prompt_template_files = {
        'sentence': args.prompt_template_file_sentence,
        'paragraph': args.prompt_template_file_paragraph,
        'chunk': args.prompt_template_file_chunk
    }

    process_document(
        args.document_file,
        args.few_shot_file,
        args.entity_file,
        prompt_template_files,
        args.output_json_file,
        args.output_csv_file,
        args.num_workers,
        args.log_prompt,
        args.chunk_size,
        args.extraction_level
    )

if __name__ == '__main__':
    main()
'python scripts/filox/entity_relation_extraction.py --document_file ./datasets/preprocessed_judgment.txt --few_shot_file ./datasets/few_shot_examples_triple.txt --entity_file ./datasets/entity_types.txt --extraction_level paragraph --prompt_template_file_chunk ./prompts/prompt_paragraph.txt --output_json_file ./output/results.json --output_csv_file ./output/results.csv --num_workers 32 --log_prompt --chunk_size 41'

