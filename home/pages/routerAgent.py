# router_agent.py
# Minimal question classifier: HEALTH vs GENERAL
# - Uses local FLAN-T5 via transformers (no paid APIs)
# - Falls back to a keyword heuristic if the model output is ambiguous

from typing import Literal, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

Label = Literal["health", "general"]

# --- Lightweight keyword safety net (edit keywords as you like) ---
_HEALTH_KWS = {
    "health","medical","medicine","clinical","disease","symptom","diagnosis","treatment",
    "dose","dosage","tablet","capsule","supplement","vitamin","nutrient","nutrition","diet",
    "protein","carb","carbohydrate","fat","cholesterol","blood pressure","hypertension",
    "diabetes","glycemic","calorie","kcal","macro","micronutrient","iron","ferritin","hemoglobin",
    "anemia","exercise","workout","fitness","bmi","allergy","intolerance","gluten","lactose",
    "pregnancy","breastfeeding","pediatric","kids","child","omega 3","fiber","probiotic","prebiotic"
}

def _kw_health(text: str, threshold: int = 2) -> bool:
    t = (text or "").lower()
    hits = sum(1 for kw in _HEALTH_KWS if kw in t)
    return hits >= threshold

# --- Build local LLM once and reuse ---
def build_local_classifier(model_name: str = "google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=8,
        temperature=0.0,
        repetition_penalty=1.0,
        device=0 if tok.backend_tokenizer.__class__.__name__ != "PreTrainedTokenizerFast" and False else -1
    )

_CLASSIFY_INSTR = """Classify the user question into exactly one label on the last line: HEALTH or GENERAL.

HEALTH = about health, medicine, symptoms, labs, nutrition, diets, supplements, or fitness.
GENERAL = everything else.

Question: {q}

Label:
""".strip()

def classify_question(
    q: str,
    clf_pipe=None
) -> Tuple[Label, Dict]:
    """
    Returns: ("health"|"general", debug_info)
    Priority: (1) FLAN-T5 label -> (2) keyword heuristic.
    """
    q = (q or "").strip()
    dbg = {"flan_raw": None, "kw_health": _kw_health(q)}

    # Try model
    if clf_pipe is not None and q:
        try:
            out = clf_pipe(_CLASSIFY_INSTR.format(q=q))[0]["generated_text"]
            dbg["flan_raw"] = out
            up = out.upper()
            if "HEALTH" in up and "GENERAL" not in up:
                return "health", dbg
            if "GENERAL" in up and "HEALTH" not in up:
                return "general", dbg
        except Exception as e:
            dbg["error"] = str(e)

    # Fallback keyword check
    return ("health" if dbg["kw_health"] else "general"), dbg


# --- Quick manual test ---
if __name__ == "__main__":
    pipe = build_local_classifier()  # load once
    while True:
        try:
            q = input("\nAsk: ").strip()
            if not q: break
            label, info = classify_question(q, clf_pipe=pipe)
            print(f"→ Detected: {label.upper()}  |  debug: {info}")
        except KeyboardInterrupt:
            break
