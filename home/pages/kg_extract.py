# kg_extract.py
import re
from typing import List, Tuple
import spacy
nlp = spacy.load("en_core_web_sm")  # make sure model is installed

Triplet = Tuple[str, str, str]

REL_PATTERNS = [
    (re.compile(r"\b(?P<h>[\w\- ]+?)\s+(?:contains|is\s+rich\s+in|has)\s+(?P<t>[\w\- ]+)\b", re.I), "CONTAINS"),
    (re.compile(r"\b(?P<h>[\w\- ]+?)\s+interacts\s+with\s+(?P<t>[\w\- ]+)\b", re.I), "INTERACTS_WITH"),
    (re.compile(r"\b(?P<h>[\w\- ]+?)\s+(?:helps|benefits|supports)\s+(?P<t>[\w\- ]+)\b", re.I), "BENEFITS"),
    (re.compile(r"\b(?P<h>[\w\- ]+?)\s+(?:treats|treatment\s+for|for\s+treatment\s+of)\s+(?P<t>[\w\- ]+)\b", re.I), "TREATMENT_FOR"),
    (re.compile(r"\b(?P<h>[\w\- ]+?)\s+(?:causes|leads\s+to)\s+(?P<t>[\w\- ]+)\b", re.I), "CAUSES"),
]

def extract_entities(text: str) -> List[str]:
    doc = nlp(text or "")
    names = [e.text.strip() for e in doc.ents if e.text and len(e.text.strip())>2]
    # simple capitalized fallback (captures things like "Vitamin D")
    names += [m.group(0) for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z0-9]+){0,2})\b", text or "")]
    uniq = []
    for x in names:
        if x not in uniq:
            uniq.append(x)
    return uniq

def extract_triples_rules(text: str) -> List[Triplet]:
    triples = []
    for pat, rel in REL_PATTERNS:
        for m in pat.finditer(text or ""):
            h = (m.group("h") or "").strip()
            t = (m.group("t") or "").strip()
            if h and t and h.lower()!=t.lower():
                triples.append((h, rel, t))
    return triples
