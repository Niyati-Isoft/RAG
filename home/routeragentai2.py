# routeragentai.py
# ----------------------------------------------------------
# Cloud-based classifier for HEALTH vs GENERAL
# Supports OpenAI or Claude depending on user selection.
# ----------------------------------------------------------

from typing import Dict, Tuple, Literal

Label = Literal["health", "general"]

# ---------- CLASSIFIER PROMPT ----------
_CLASSIFY_PROMPT = """
You must classify the user's question into one of two categories:

HEALTH  → medical, symptoms, diagnostics, treatments, diet, nutrition, calories,
           macros/micros, supplements, exercise affecting health.

GENERAL → all other topics: technology, finance, shopping, food labels, prices,
           packaging, travel, coding, regulations, general product info.

RULES:
- If unsure → choose GENERAL.
- Output ONLY one word: HEALTH or GENERAL.

Question:
{q}

Label:
""".strip()


# ----------------------------------------------------------------
# 1) OPENAI classifier
# ----------------------------------------------------------------
def classify_openai(q: str, client) -> Tuple[Label, Dict]:
    dbg = {}

    prompt = _CLASSIFY_PROMPT.format(q=q)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        temperature=0
    )

    text = resp.choices[0].message["content"].strip().upper()
    dbg["raw"] = text

    if "HEALTH" in text and "GENERAL" not in text:
        return "health", dbg
    return "general", dbg
        

# ----------------------------------------------------------------
# 2) CLAUDE classifier
# ----------------------------------------------------------------
def classify_claude(q: str, client) -> Tuple[Label, Dict]:
    dbg = {}

    prompt = _CLASSIFY_PROMPT.format(q=q)

    resp = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    text = resp.content[0].text.strip().upper()
    dbg["raw"] = text

    if "HEALTH" in text and "GENERAL" not in text:
        return "health", dbg
    return "general", dbg
