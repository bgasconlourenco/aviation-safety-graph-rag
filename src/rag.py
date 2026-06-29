"""Phase 7: RAG retrieval — vector search → graph traversal → LLM answer."""

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver
from sentence_transformers import SentenceTransformer

from src.embed import MODEL_NAME

load_dotenv()

TOP_K = 5
OLLAMA_MODEL = "gemma3"

# --- Cypher ---

_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('incident_embedding', $k, $query_vector)
YIELD node AS i, score
RETURN i.acn      AS acn,
       i.synopsis AS synopsis,
       i.narrative AS narrative,
       i.date     AS date,
       i.year     AS year,
       score
ORDER BY score DESC
"""

_ENRICH = """
MATCH (i:Incident {acn: $acn})
OPTIONAL MATCH (i)-[:HAD_CONDITIONS]->(fc:FlightCondition)
OPTIONAL MATCH (i)-[:OCCURRED_IN]->(lc:LightCondition)
OPTIONAL MATCH (i)-[:WAS_MISSION]->(mt:MissionType)
OPTIONAL MATCH (i)-[:HAS_PRIMARY_PROBLEM]->(pp:ProblemType)
OPTIONAL MATCH (i)-[:DURING_PHASE]->(fp:FlightPhase)
OPTIONAL MATCH (i)-[:INVOLVED_ANOMALY]->(a:AnomalyType)
RETURN fc.name          AS flight_conditions,
       lc.name          AS light,
       mt.name          AS mission,
       pp.name          AS primary_problem,
       fp.name          AS flight_phase,
       collect(a.name)  AS anomalies
"""


# --- Core pipeline steps ---

def retrieve(
    driver: Driver,
    query_text: str,
    model: SentenceTransformer | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Embed *query_text* and return the top-k most similar incidents."""
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(query_text).tolist()
    with driver.session() as session:
        result = session.run(_VECTOR_SEARCH, k=top_k, query_vector=query_vector)
        return [dict(r) for r in result]


def enrich(driver: Driver, acns: list[str]) -> dict[str, dict]:
    """Fetch graph-connected metadata for each ACN."""
    enriched: dict[str, dict] = {}
    with driver.session() as session:
        for acn in acns:
            record = session.run(_ENRICH, acn=acn).single()
            if record:
                enriched[acn] = dict(record)
    return enriched


def _format_context(hits: list[dict], enrichment: dict[str, dict]) -> str:
    """Build a human-readable context block from vector hits + graph metadata."""
    parts = []
    for idx, hit in enumerate(hits, 1):
        acn = hit["acn"]
        meta = enrichment.get(acn, {})
        lines = [f"[Incident {idx}] ACN {acn}  (similarity {hit['score']:.3f})"]

        if hit.get("date"):
            lines.append(f"  Date: {hit['date']}")
        if hit.get("synopsis"):
            lines.append(f"  Synopsis: {hit['synopsis']}")
        if hit.get("narrative"):
            text = str(hit["narrative"])
            trimmed = text[:600] + ("…" if len(text) > 600 else "")
            lines.append(f"  Narrative: {trimmed}")

        if meta.get("mission"):
            lines.append(f"  Mission: {meta['mission']}")
        if meta.get("flight_phase"):
            lines.append(f"  Flight phase: {meta['flight_phase']}")
        if meta.get("flight_conditions"):
            lines.append(f"  Flight conditions: {meta['flight_conditions']}")
        if meta.get("light"):
            lines.append(f"  Light: {meta['light']}")
        if meta.get("primary_problem"):
            lines.append(f"  Primary problem: {meta['primary_problem']}")
        if meta.get("anomalies"):
            lines.append(f"  Anomalies: {', '.join(meta['anomalies'])}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are an aviation safety analyst with deep expertise in helicopter operations. "
        "Use the NASA ASRS incident reports below to answer the question. "
        "Cite relevant ACN numbers when appropriate.\n\n"
        f"--- INCIDENT REPORTS ---\n{context}\n--- END REPORTS ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def answer(question: str, context: str) -> str:
    """Generate an answer via Ollama (primary) or the Claude API (fallback)."""
    prompt = _build_prompt(question, context)

    # --- Try Ollama first ---
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    try:
        import ollama  # noqa: PLC0415

        client = ollama.Client(host=ollama_url)
        response = client.generate(model=ollama_model, prompt=prompt)
        return response["response"].strip()
    except Exception:  # noqa: BLE001
        pass

    # --- Fall back to Claude API ---
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM available: Ollama not reachable and ANTHROPIC_API_KEY is not set."
        )

    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
        for block in reversed(msg.content):
            if block.type == "text":
                return block.text.strip()
        return ""


# --- Top-level entry point ---

def ask(
    driver: Driver,
    question: str,
    model: SentenceTransformer | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Full RAG pipeline: embed → retrieve → enrich → answer.

    Returns a dict with keys: question, answer, sources, context.
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    hits = retrieve(driver, question, model=model, top_k=top_k)
    acns = [h["acn"] for h in hits]
    enrichment = enrich(driver, acns)
    context = _format_context(hits, enrichment)
    response = answer(question, context)

    return {
        "question": question,
        "answer": response,
        "sources": hits,
        "enrichment": enrichment,
        "context": context,
    }


if __name__ == "__main__":
    from src.graph_load import connect

    driver = connect()
    result = ask(driver, "What are common causes of helicopter accidents in IMC?")
    print(result["answer"])
    print("\n--- Sources ---")
    for src in result["sources"]:
        print(f"  ACN {src['acn']}  score={src['score']:.3f}  {src.get('synopsis', '')[:80]}")
    driver.close()
