# ===============================================================
# agent_health.py — Health RAG Agent
# ===============================================================

from rag_core import run_rag_pipeline

HEALTH_SYSTEM = """
You are a health information assistant.
You MUST use only the retrieved context.
Do not diagnose diseases. Do not claim treatments.
Always give safe, general educational information.
"""

def run_health_rag(question: str, llm_engine: str,
                   openai_client=None, claude_client=None):

    return run_rag_pipeline(
        question=question,
        system_prompt=HEALTH_SYSTEM,
        llm_engine=llm_engine,
        openai_client=openai_client,
        claude_client=claude_client
    )
