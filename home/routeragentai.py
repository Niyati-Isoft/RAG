# router_agent.py
# -------------------------------------------------------------------
# LangGraph Router Agent:
#   - Uses local FLAN-T5 to classify a question as HEALTH or GENERAL
#   - Then routes to a tool:
#         health_tool(question)  -> answer string
#         general_tool(question) -> answer string
#   - No keyword list, no heuristics – pure LLM classification
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Literal, TypedDict, Dict, Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langgraph.graph import StateGraph, END

Label = Literal["health", "general"]


# -------------------------------------------------------------------
# 1. System prompt for classification (agent's "brain" for routing)
# -------------------------------------------------------------------

_CLASSIFY_INSTR = """
You are a strict binary classifier for user questions.

Your task: decide whether the question is about **HEALTH** or **GENERAL** topics.

Definitions (be strict):

- HEALTH = questions primarily about:
  • human or animal health, medicine, symptoms, diagnoses, treatments, side effects  
  • laboratory tests, blood reports, biomarkers, medical conditions  
  • nutrition, diet quality, macros/micros, calories, weight loss/gain, supplements  
  • exercise or fitness in a health / body context

- GENERAL = **everything else**, including:
  • prices, legal rules, regulations, labelling, marketing, packaging  
  • general food/product info not tied to health effects  
  • technology, finance, education, travel, coding, etc.

Special rules (important):

1. If the question is SHORT or VAGUE (e.g. 1–3 generic words like
   "hello", "general question"), classify it as GENERAL
   unless it clearly contains medical/health/nutrition terms.
2. Questions about food **labelling, packaging, marketing, or regulations**
   should be GENERAL unless they explicitly ask about health impact
   (e.g. “Is this healthy for me?”, “Is this safe for diabetics?”).
3. When you are UNSURE, you MUST choose GENERAL. Do **not** guess HEALTH.
4. Output **exactly one word** on the final line: either HEALTH or GENERAL,
   with no punctuation or extra text.

Question:
{q}

Label:
""".strip()



# -------------------------------------------------------------------
# 2. Build local FLAN-T5 classifier (once, and reuse)
# -------------------------------------------------------------------

def build_local_classifier(model_name: str = "google/flan-t5-base"):
    """
    Returns a HuggingFace pipeline for text2text-generation using FLAN-T5.
    This is used ONLY for classification (HEALTH vs GENERAL).
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    clf = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=8,
        temperature=0.0,
        repetition_penalty=1.0,
        device=-1,   # CPU; change to 0 if you want GPU
    )
    return clf


def classify_question_llm(
    q: str,
    clf_pipe,
) -> tuple[Label, Dict[str, Any]]:
    """
    Pure LLM-based classifier.
    Returns: (label, debug_info)

    label: "health" or "general"
    debug_info: {"flan_raw": full_model_output}
    """
    q = (q or "").strip()
    dbg: Dict[str, Any] = {"flan_raw": None}

    if not q:
        # empty → treat as GENERAL by default
        return "general", dbg

    prompt = _CLASSIFY_INSTR.format(q=q)
    out = clf_pipe(prompt)[0]["generated_text"]
    dbg["flan_raw"] = out

    up = out.strip().upper()

    # Be strict: look only at the last token-ish
    if "HEALTH" in up and "GENERAL" not in up:
        return "health", dbg
    if "GENERAL" in up and "HEALTH" not in up:
        return "general", dbg

    # Ambiguous → safe default
    return "general", dbg


# -------------------------------------------------------------------
# 3. LangGraph state definition
# -------------------------------------------------------------------

class RouterState(TypedDict, total=False):
    """
    State that flows through the LangGraph router agent.
    """
    question: str
    label: Label
    answer: str
    debug: Dict[str, Any]


# -------------------------------------------------------------------
# 4. Graph nodes (classify + call tools)
# -------------------------------------------------------------------

def node_classify(state: RouterState, clf_pipe) -> RouterState:
    """
    Node 1: use FLAN-T5 to decide HEALTH vs GENERAL.
    """
    q = (state.get("question") or "").strip()
    label, dbg = classify_question_llm(q, clf_pipe=clf_pipe)
    new_state: RouterState = {
        "question": q,
        "label": label,
        "debug": dbg,
    }
    return new_state


def node_health_tool(state: RouterState, health_tool: Callable[[str], str]) -> RouterState:
    """
    Node 2a: call the health-specific tool.
    health_tool: function(question:str) -> answer:str
    """
    q = state["question"]
    ans = health_tool(q)
    new_state: RouterState = {
        "question": q,
        "label": "health",
        "answer": ans,
        "debug": state.get("debug", {}),
    }
    return new_state


def node_general_tool(state: RouterState, general_tool: Callable[[str], str]) -> RouterState:
    """
    Node 2b: call the general tool.
    general_tool: function(question:str) -> answer:str
    """
    q = state["question"]
    ans = general_tool(q)
    new_state: RouterState = {
        "question": q,
        "label": "general",
        "answer": ans,
        "debug": state.get("debug", {}),
    }
    return new_state


# -------------------------------------------------------------------
# 5. Edge router: decide which branch after classification
# -------------------------------------------------------------------

def router_edge(state: RouterState) -> str:
    """
    Reads state["label"] and decides which node to go to.
    Must match the keys used in add_conditional_edges.
    """
    label: Label = state.get("label", "general")
    return "health_tool" if label == "health" else "general_tool"


# -------------------------------------------------------------------
# 6. Build the LangGraph router app (agent)
# -------------------------------------------------------------------

def build_router_agent(
    health_tool: Callable[[str], str],
    general_tool: Callable[[str], str],
    clf_pipe=None,
):
    """
    Builds and compiles a LangGraph app that:
      1. Classifies the question with FLAN-T5
      2. Routes to health_tool or general_tool
      3. Returns final state with answer + debug

    Parameters
    ----------
    health_tool: function(question:str) -> str
        Your health RAG or health-answer function.
    general_tool: function(question:str) -> str
        Your general RAG or general-answer function.
    clf_pipe: optional HuggingFace pipeline for FLAN-T5.
        If None, build_local_classifier() will be used.
    """
    if clf_pipe is None:
        clf_pipe = build_local_classifier()

    graph = StateGraph(RouterState)

    # Wrap nodes so they can capture the external objects
    graph.add_node(
        "classify",
        lambda s: node_classify(s, clf_pipe=clf_pipe),
    )
    graph.add_node(
        "health_tool",
        lambda s: node_health_tool(s, health_tool=health_tool),
    )
    graph.add_node(
        "general_tool",
        lambda s: node_general_tool(s, general_tool=general_tool),
    )

    graph.set_entry_point("classify")

    # Conditional edge leaving "classify"
    graph.add_conditional_edges(
        "classify",
        router_edge,
        {
            "health_tool": "health_tool",
            "general_tool": "general_tool",
        },
    )

    graph.add_edge("health_tool", END)
    graph.add_edge("general_tool", END)

    app = graph.compile()
    return app


# -------------------------------------------------------------------
# 7. Convenience helper: run the agent on one question
# -------------------------------------------------------------------

def route_and_answer(
    question: str,
    health_tool: Callable[[str], str],
    general_tool: Callable[[str], str],
    clf_pipe=None,
) -> RouterState:
    """
    One-shot helper:
      - builds the router agent
      - runs it on the given question
      - returns the final RouterState
    """
    app = build_router_agent(
        health_tool=health_tool,
        general_tool=general_tool,
        clf_pipe=clf_pipe,
    )
    init_state: RouterState = {"question": question}
    final_state: RouterState = app.invoke(init_state)
    return final_state


# -------------------------------------------------------------------
# 8. Quick manual test (CLI)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy tools just to see routing behaviour.
    # In your real app, replace these with your RAG functions.
    def health_tool_fn(q: str) -> str:
        return f"[HEALTH TOOL] I will answer this health-related question: {q}"

    def general_tool_fn(q: str) -> str:
        return f"[GENERAL TOOL] I will answer this general question: {q}"

    clf = build_local_classifier()

    print("Router agent ready. Type a question (blank line to quit):")
    while True:
        try:
            q = input("\nQ: ").strip()
            if not q:
                break
            state = route_and_answer(
                question=q,
                health_tool=health_tool_fn,
                general_tool=general_tool_fn,
                clf_pipe=clf,
            )
            print(f"Label:   {state.get('label','?').upper()}")
            print(f"Answer:  {state.get('answer','')}")
            print(f"Debug:   {state.get('debug',{})}")
        except KeyboardInterrupt:
            break


#checking