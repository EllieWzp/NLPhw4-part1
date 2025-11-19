import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import string

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


###### YOUR CODE BEGINS HERE ######
_detok = TreebankWordDetokenizer()

# QWERTY keyboard neighbors for minor typos
_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
    'f': 'rtgdvc', 'g': 'tyfhvb', 'h': 'yugjbn', 'i': 'uokj', 'j': 'uikhmn',
    'k': 'ijolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'tfde', 's': 'wedxz', 't': 'ygfr', 'u': 'yhji',
    'v': 'cfgb', 'w': 'qes', 'x': 'zsdc', 'y': 'tugh', 'z': 'asx'
}

def _rand_typo(word):
    """Randomly injects a typo by swapping, deleting, or replacing with neighbor key."""
    if len(word) <= 3:
        return word
    op = random.choice(["neighbor", "delete", "swap"])
    chars = list(word)
    i = random.randrange(len(chars))
    if op == "neighbor":
        c = chars[i].lower()
        if c in _NEIGHBORS and _NEIGHBORS[c]:
            chars[i] = random.choice(_NEIGHBORS[c])
    elif op == "delete" and len(chars) > 3:
        del chars[i]
    elif op == "swap" and len(chars) > 3:
        j = i if i < len(chars) - 1 else i - 1
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
    return "".join(chars)

def _get_synonym(word):
    """Returns a near-length synonym from WordNet."""
    syns = wordnet.synsets(word)
    cands = set()
    for s in syns:
        for lemma in s.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower() and re.fullmatch(r"[A-Za-z\-']+", w):
                cands.add(w)
    if not cands:
        return None
    cands = sorted(cands, key=lambda w: abs(len(w) - len(word)))
    return random.choice(cands[:3])

def custom_transform(example):
    """
    Applies a 'reasonable' OOD transformation:
    - random synonym replacement (p=0.15)
    - random keyboard typo (p=0.07)
    - occasional punctuation noise (p=0.1)
    Keeps the sentiment label unchanged.
    """
    text = example["text"]
    tokens = word_tokenize(text)
    new_tokens = []
    for tok in tokens:
        new_tok = tok
        # Skip punctuation, numbers, or very short words
        if len(tok) <= 2 or tok.isdigit() or all(ch in string.punctuation for ch in tok):
            new_tokens.append(tok)
            continue

        # 1️⃣ synonym replacement
        if random.random() < 0.25 and tok.isalpha():
            syn = _get_synonym(tok.lower())
            if syn:
                if tok.istitle():
                    syn = syn.title()
                elif tok.isupper():
                    syn = syn.upper()
                new_tok = syn

        # 2️⃣ typo noise
        if random.random() < 0.12 and tok.isalpha():
            new_tok = _rand_typo(new_tok)

        new_tokens.append(new_tok)

    # 3️⃣ optional punctuation injection
    if random.random() < 0.2:
        pos = random.randint(0, len(new_tokens))
        new_tokens.insert(pos, random.choice(["!", "?", "…"]))

    example["text"] = _detok.detokenize(new_tokens)
    return example
###### YOUR CODE ENDS HERE ######
