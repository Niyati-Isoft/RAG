# ===============================================================
# router_orchestrator.py
# Router + Orchestrator (one file)
# ===============================================================

from typing import Tuple, Dict, Any, Literal
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

Label = Literal["health", "general"]

# ---------- Router Prompt ----------
_ROUTE_PROMPT = """
You MUST classify the question into exactly one category:

HEALTH → anything about:
- medical conditions, symptoms, treatments
- nutrition, diet, calories, macros, vitamins
- supplements, BMI, blood tests, physiology

GENERAL → everything else:
- technology, coding, finance, pricing, business
- shopping, products, travel, education, math

If uncertain → GENERAL

Question:
{q}

Output only: HEALTH or GENERAL
"""

# ---------- FLAN Router ----------
def build_flan_router():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tok)

def classify_flan(q, clf):
    out = clf(_ROUTE_PROMPT.format(q=q))[0]["generated_text"].strip().upper()
    if "HEALTH" in out and "GENERAL" not in out:
        return "health", {"raw": out}
    return "general", {"raw": out}

# ---------- OpenAI Router ----------
def classify_openai(q, client):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(q=q)}],
            max_tokens=5, temperature=0
        )
        out = r.choices[0].message.content.strip().upper()
    except Exception as e:
        return "general", {"error": str(e)}

    if "HEALTH" in out and "GENERAL" not in out:
        return "health", {"raw": out}
    return "general", {"raw": out}

# ---------- Claude Router ----------
def classify_claude(q, client):
    try:
        r = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": _ROUTE_PROMPT.format(q=q)}]
        )
        out = r.content[0].text.strip().upper()
    except Exception as e:
        return "general", {"error": str(e)}

    if "HEALTH" in out and "GENERAL" not in out:
        return "health", {"raw": out}
    return "general", {"raw": out}


# ---------- Orchestrator ----------
def orchestrate(question: str,
                router_engine: str,
                answer_engine: str,
                flan_router=None,
                openai_client=None,
                claude_client=None):

    # 1) CLASSIFY
    if router_engine == "openai":
        label, dbg = classify_openai(question, openai_client)

    elif router_engine == "claude":
        label, dbg = classify_claude(question, claude_client)

    else:
        label, dbg = classify_flan(question, flan_router)

    # 2) CALL CORRECT AGENT
    if label == "health":
        from agent_health import run_health_rag
        ans = run_health_rag(question, answer_engine, openai_client, claude_client)

    else:
        from agent_general import run_general_rag
        ans = run_general_rag(question, answer_engine, openai_client, claude_client)

    return {
        "label": label,
        "router_debug": dbg,
        "answer": ans
    }
