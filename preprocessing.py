from datasketch import MinHash, MinHashLSH
# from langdetect import detect
# from bs4 import BeautifulSoup
import re
import shutil
from collections import Counter
from pathlib import Path

def copy_metadata_files(input_directory: Path, output_directory: Path) -> None:
    for metadata_path in input_directory.rglob("metadata.json"):
        relative_path = metadata_path.relative_to(input_directory)
        output_path = output_directory / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, output_path)
    print(f"Copied metadata.json files from {input_directory} to {output_directory}.")

def minhash_deduplication(texts, threshold=0.7):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for i, doc in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in set(doc.split()):
            m.update(word.encode('utf8'))
        if not lsh.query(m):
            lsh.insert(f"doc{i}", m)
            unique_texts.append(doc)
    return unique_texts

def clean_html_and_filter_lang(texts, lang='en'):
    filtered = []
    for txt in texts:
        txt = BeautifulSoup(txt, 'html.parser').get_text()
        try:
            if detect(txt.strip()) == lang:
                filtered.append(txt.strip())
        except:
            continue
    return filtered

def strip_pii(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    text = re.sub(r'\b\d{12,19}\b', '[CREDIT_CARD]', text)
    text = re.sub(r'\b(?:\d{3}-){2}\d{4}\b', '[PHONE]', text)
    return text

def remove_repetitive_ngrams(text, n=3, threshold=3):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    counts = Counter(ngrams)
    repetitive = [ngram for ngram, count in counts.items() if count >= threshold]

    for phrase in repetitive:
        escaped_phrase = re.escape(phrase)
        text = re.sub(rf'(?:{escaped_phrase}\s*){{{threshold},}}', phrase + ' ', text)

    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def preprocess(text) :
    # chunking text
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    original_chunks = len(chunks)
    # minhash deduplication
    step2 = minhash_deduplication(chunks)
    dedup_removed = original_chunks - len(step2)
    
    # PII stripping and repetitive n-gram removal
    pii_replacements = 0
    cleaned_data = []

    for chunk in step2:
        before = chunk
        step3 = strip_pii(chunk)

        if step3 != before:
            pii_replacements += 1

        step4 = remove_repetitive_ngrams(step3)
        cleaned_data.append(step4)
    
    cleaned_text = "\n\n".join(cleaned_data)
    
    print(f"Paragraphs: {original_chunks}")
    print(f"Duplicates removed: {dedup_removed}")
    print(f"PII replacements: {pii_replacements}")
    
    print("Original length:", len(text))
    print("Processed length:", len(cleaned_text))
    
    return cleaned_text