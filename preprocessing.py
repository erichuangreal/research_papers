from datasketch import MinHash, MinHashLSH
# from langdetect import detect
# from bs4 import BeautifulSoup
import re
from collections import Counter

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
        # regex-safe version of the phrase
        escaped_phrase = re.escape(phrase)
        # match the phrase repeated 2+ times with optional whitespace
        text = re.sub(rf'(?:{escaped_phrase}\s*){{{threshold},}}', phrase + ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def preprocess(text) :
    # step1 = clean_html_and_filter_lang(text)
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    step2 = minhash_deduplication(chunks)
    step3 = [strip_pii(t) for t in step2]
    cleaned_data = [remove_repetitive_ngrams(t) for t in step3]
    
    return "\n\n".join(cleaned_data)