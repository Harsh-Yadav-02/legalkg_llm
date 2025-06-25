import argparse
import os
import re
import json
import nltk
import numpy as np
import regex
import sys
from collections import OrderedDict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from num2words import num2words
import spacy

# Uncomment these if NLTK resources are not already downloaded.
# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("averaged_perceptron_tagger", quiet=True)
# nltk.download("wordnet", quiet=True)

class ImproveParaProcessor:
    def __init__(self):
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        # self.nlp = en_core_web_trf.load()
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = {*stopwords.words("english")}
        self.dates = []
        self.unique_ids = []
        # By default, all processing steps are enabled.
        self.options = {
            'remove_whitespaces': True,
            'remove_footers': True,
            'expand_contractions': True,
            'remove_patterns': True,
            'remove_special_chars': True,
            'remove_stop_words': True,
            'convert_numbers_to_words': True,
            'convert_to_lowercase': True,
            'remove_abbreviations': True,
            'lemmatize': True
        }
    
    def load(self, paragraph):
        # sentences = self.tokenizer.tokenize(paragraph)
        doc = self.nlp(paragraph)
        sentences = [sent.text for sent in doc.sents]
        words = [
            word for sentence in sentences for word in word_tokenize(sentence)
            if word.lower() not in self.stopwords and regex.search(r"(?ui)\p{L}+", word)
        ]
        return sentences, words

    def get_list_of_dates_ids(self, paragraph):
        date_pattern = r'\b\d{1,2}[/.]\d{1,2}[/.]\d{4}\b'
        pattern1 = re.compile(r'[A-Za-z]+\.[A-Za-z]+\.\s*\([A-Za-z.\-/]+\)\s\d+\s*/\s*\d+')
        pattern2 = re.compile(r'[A-Za-z]+[A-Za-z]+\s*\([A-Za-z\-/]+\)\s\d+\s*/\s*\d+')
        unique_ids_pattern = re.compile(f'{pattern1.pattern}|{pattern2.pattern}')
        self.dates = re.findall(date_pattern, paragraph)
        self.unique_ids = re.findall(unique_ids_pattern, paragraph)

    def remove_stop_words(self, paragraph):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(paragraph)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    def remove_patterns(self, paragraph):
        paragraph = re.sub(r'\n+', ' ', paragraph)
        return paragraph
   
    def remove_whitespaces(self, paragraph):
        return re.sub(r'\s+', ' ', paragraph)

    def remove_footers(self, paragraph):
        pattern1 = r'[A-Za-z]+\.*[A-Za-z]+\.*\s*\([A-Za-z.\-/]+\)\s([A-Za-z]+\.)?(&*\s*\d+/\d+;?\s*)*'
        pattern2 = r'[A-Za-z]+[A-Za-z]+\s*\([A-Za-z\-/]+\)\s\d+\s*/\s*\d+\s*'
        page_pattern = r'\s*\n*Page\s*\d+(\s+of\s+\d+)?'
        combined_pattern = f'({pattern1}|{pattern2}){page_pattern}'
        footer_pattern = re.compile(combined_pattern)
        return re.sub(footer_pattern, '', paragraph)
        
    def remove_abbreviations(self, paragraph):
        # Assumes a JSON file 'abbreviations.json' exists in the specified directory.
        with open(os.path.join(os.getcwd(), './datasets/abbreviations.json'), 'r') as f:
            abbrevs_dict = json.load(f)
        for key, value in abbrevs_dict.items():
            # Remove any trailing period from the key for a consistent match,
            # then match an optional period only if it is immediately followed by whitespace or end-of-string.
            pattern = r'\b' + re.escape(key.rstrip('.')) + r'\.?(?=\s|$)'
            paragraph = re.sub(pattern, value, paragraph)
        return paragraph

    def expand_contractions(self, data):
        # Assumes a file 'contraction_map.txt' exists in the specified directory.
        with open(os.path.join(os.getcwd(), './datasets/contraction_map.txt')) as f:
            CONTRACTION_MAP = f.read()
        js = json.loads(CONTRACTION_MAP)
        tokens = data.split(' ')
        tokens = [js.get(word, word) for word in tokens]
        return ' '.join(tokens)

    def remove_special_chars(self, text):
        symbols = "!“”\"#$%&*+-,/:;<=>?@[\]^_`{|}~‘’"
        for sym in symbols:
            text = np.char.replace(text, sym, ' ')
            text = np.char.replace(text, "  ", " ")
        text = str(text)
        for date in self.dates:
            text = text.replace(date.replace('/', ' ').replace('.', ' '), date)
        for uid in self.unique_ids:
            id_replaced = uid.replace('/', ' ').replace('-', ' ')
            text = text.replace(id_replaced, uid, 1)
        return text
    
    def lemmatization(self, data):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(data)
        pos_tags = nltk.pos_tag(tokens)
        new_text = ""
        for token, tag in pos_tags:
            if token not in [",", ".", "!", "?", ";", ":"]:
                if tag.startswith('N'):
                    new_text += " " + lemmatizer.lemmatize(token, 'n')
                elif tag.startswith('V'):
                    new_text += " " + lemmatizer.lemmatize(token, 'v')
                elif tag.startswith('R'):
                    new_text += " " + lemmatizer.lemmatize(token, 'r')
                elif tag.startswith('J'):
                    new_text += " " + lemmatizer.lemmatize(token, 'a')
                else:
                    new_text += " " + lemmatizer.lemmatize(token)
            else:
                new_text += token
        return new_text.strip()

    def convert_numbers_to_words(self, paragraph):
        paragraph = str(paragraph)
        unique_ids_dates = self.dates + self.unique_ids
        paragraph = re.sub(r'(\d),(\d)', r'\1\2', paragraph)
        for i, uid in enumerate(unique_ids_dates):
            paragraph = paragraph.replace(uid, f'{{placeholder{i}}}')
        paragraph = re.sub(r'\b\d+\b', lambda x: num2words(int(x.group())), paragraph)
        for i, uid in enumerate(unique_ids_dates):
            paragraph = paragraph.replace(f'{{placeholder{i}}}', uid)
        return paragraph.replace('-', ' ')
    
    def convert_to_lowercase(self, data):
        return data.lower()

    def decision(self, paragraph):
        self.get_list_of_dates_ids(paragraph)
        imp_paragraph = paragraph
        if self.options.get('remove_whitespaces'):
            imp_paragraph = self.remove_whitespaces(imp_paragraph)
        if self.options.get('remove_footers'):
            imp_paragraph = self.remove_footers(imp_paragraph)
        if self.options.get('expand_contractions'):
            imp_paragraph = self.expand_contractions(imp_paragraph)
        if self.options.get('remove_patterns'):
            imp_paragraph = self.remove_patterns(imp_paragraph)
        if self.options.get('remove_special_chars'):
            imp_paragraph = self.remove_special_chars(imp_paragraph)
        if self.options.get('remove_stop_words'):
            imp_paragraph = self.remove_stop_words(imp_paragraph)
        if self.options.get('convert_numbers_to_words'):
            imp_paragraph = self.convert_numbers_to_words(imp_paragraph)
        if self.options.get('convert_to_lowercase'):
            imp_paragraph = self.convert_to_lowercase(imp_paragraph)
        if self.options.get('remove_abbreviations'):
            imp_paragraph = self.remove_abbreviations(imp_paragraph)
        if self.options.get('lemmatize'):
            imp_paragraph = self.lemmatization(imp_paragraph)
        return OrderedDict([('imp_content', imp_paragraph)])

def process_document(input_file, output_dir=None, processor=None):
    if processor is None:
        processor = ImproveParaProcessor()
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # Assuming paragraphs are separated by double newlines.
    paragraphs = content.split('\n\n')
    processed_paras = []
    for para in paragraphs:
        if para.strip():
            proc = processor.decision(para)
            processed_paras.append(proc['imp_content'])
        else:
            processed_paras.append('')
   # Join processed paragraphs with two newlines
    new_content = "\n\n".join(processed_paras)

    # Protect newlines that are followed by (optionally after spaces) an integer and a period.
    # This placeholder token (<<<NEWPARA>>>) marks new paragraphs that should not be replaced.
    protected_content = re.sub(r'\n+(?=\s*\d+\.)', '<<<NEWPARA>>>', new_content)

    # Remove extra whitespace (including newlines) by replacing any sequence of whitespace with a single space.
    text_for_tokenization = re.sub(r'\s+', ' ', protected_content)

    # Restore the protected newline markers.
    text_for_tokenization = text_for_tokenization.replace('<<<NEWPARA>>>', '\n')

    # Tokenize the cleaned text into sentences using NLTK.
    # sentences = sent_tokenize(text_for_tokenization)
    doc = processor.nlp(text_for_tokenization)
    sentences = [sent.text for sent in doc.sents]

    # Optionally, join sentences with a newline to output one sentence per line.
    final_content = "\n".join(sentences)

    if not output_dir:
        output_dir = os.path.dirname(input_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_filename = "preprocessed_" + os.path.basename(input_file)
    output_path = os.path.join(output_dir, new_filename)
    with open(output_path, 'w', encoding='utf-8') as outf:
        outf.write(final_content)
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a document using selected command-line options."
    )
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory (if not provided, uses the input file's directory)")
    # Define flags for each processing step.
    parser.add_argument("--ws", action="store_true", help="Remove whitespaces")
    parser.add_argument("--footers", action="store_true", help="Remove footers")
    parser.add_argument("--contractions", action="store_true", help="Expand contractions")
    parser.add_argument("--patterns", action="store_true", help="Remove patterns")
    parser.add_argument("--special", action="store_true", help="Remove special characters")
    parser.add_argument("--stop", action="store_true", help="Remove stop words")
    parser.add_argument("--numbers", action="store_true", help="Convert numbers to words")
    parser.add_argument("--lower", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--abbrev", action="store_true", help="Remove abbreviations")
    parser.add_argument("--lemmatize", action="store_true", help="Lemmatize tokens")
    
    args = parser.parse_args()
    
    # Create a processor instance.
    processor = ImproveParaProcessor()
    
    # If any specific flags are provided, disable all default options
    # and enable only the specified ones.
    flag_keys = ['ws', 'footers', 'contractions', 'patterns', 'special', 
                 'stop', 'numbers', 'lower', 'abbrev', 'lemmatize']
    flags = vars(args)
    if any(flags[key] for key in flag_keys):
        for opt in processor.options:
            processor.options[opt] = False
        mapping = {
            'ws': 'remove_whitespaces',
            'footers': 'remove_footers',
            'contractions': 'expand_contractions',
            'patterns': 'remove_patterns',
            'special': 'remove_special_chars',
            'stop': 'remove_stop_words',
            'numbers': 'convert_numbers_to_words',
            'lower': 'convert_to_lowercase',
            'abbrev': 'remove_abbreviations',
            'lemmatize': 'lemmatize'
        }
        for key, opt in mapping.items():
            if flags[key]:
                processor.options[opt] = True
    
    output_path = process_document(args.input_file, args.output_dir, processor)
    print("Processed file saved as:", output_path)

if __name__ == "__main__":
    main()
#python scripts/filox/preprocessing.py ./datasets/judgment.txt ./datasets --contractions --abbrev
