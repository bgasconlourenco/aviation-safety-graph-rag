import os

import pytest

from src.rag import _build_prompt, _format_context

# --- Pure-Python helpers (always run) ---

_HITS = [
    {
        "acn": "1234567",
        "synopsis": "Pilot lost visual reference in fog.",
        "narrative": "The helicopter entered IMC unexpectedly while cruising at 500 ft AGL.",
        "date": "2022-03-15",
        "year": 2022,
        "score": 0.91,
    },
    {
        "acn": "9876543",
        "synopsis": "Near miss at low altitude.",
        "narrative": None,
        "date": None,
        "year": None,
        "score": 0.75,
    },
]

_ENRICHMENT = {
    "1234567": {
        "flight_conditions": "IMC",
        "light": "Night",
        "mission": "EMS",
        "primary_problem": "Human Factors",
        "flight_phase": "Cruise",
        "anomalies": ["Controlled Flight Into Terrain", "Loss of Situational Awareness"],
    },
    "9876543": {
        "flight_conditions": None,
        "light": "Day",
        "mission": "Training",
        "primary_problem": None,
        "flight_phase": None,
        "anomalies": [],
    },
}


def test_format_context_contains_acns():
    ctx = _format_context(_HITS, _ENRICHMENT)
    assert "1234567" in ctx
    assert "9876543" in ctx


def test_format_context_contains_synopsis():
    ctx = _format_context(_HITS, _ENRICHMENT)
    assert "fog" in ctx


def test_format_context_contains_meta():
    ctx = _format_context(_HITS, _ENRICHMENT)
    assert "EMS" in ctx
    assert "IMC" in ctx
    assert "Controlled Flight Into Terrain" in ctx


def test_format_context_skips_none_fields():
    ctx = _format_context(_HITS, _ENRICHMENT)
    # second incident has no primary_problem or flight_phase — those labels must not appear
    incident2_block = ctx[ctx.index("9876543"):]
    assert "Primary problem:" not in incident2_block
    assert "Flight phase:" not in incident2_block


def test_format_context_truncates_long_narrative():
    long_hit = dict(_HITS[0])
    long_hit["narrative"] = "x" * 1000
    ctx = _format_context([long_hit], {})
    # narrative trimmed to 600 chars + ellipsis
    assert "…" in ctx
    # raw 1000-char string should not appear
    assert "x" * 700 not in ctx


def test_format_context_empty_hits():
    assert _format_context([], {}) == ""


def test_build_prompt_contains_question():
    prompt = _build_prompt("Why do helicopters crash?", "some context")
    assert "Why do helicopters crash?" in prompt


def test_build_prompt_contains_context():
    prompt = _build_prompt("question", "SPECIAL_CONTEXT_STRING")
    assert "SPECIAL_CONTEXT_STRING" in prompt


def test_build_prompt_has_system_framing():
    prompt = _build_prompt("q", "c")
    assert "aviation safety" in prompt.lower()


# --- Integration tests (skip if Neo4j or embedding model unavailable) ---

@pytest.fixture(scope="module")
def driver():
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        pytest.skip("NEO4J_URI / NEO4J_PASSWORD not set")
    from src.graph_load import connect
    d = connect()
    yield d
    d.close()


@pytest.fixture(scope="module")
def embed_model():
    try:
        from sentence_transformers import SentenceTransformer

        from src.embed import MODEL_NAME
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        pytest.skip(f"sentence-transformers model unavailable: {exc}")


def test_retrieve_returns_list(driver, embed_model):
    from src.rag import retrieve
    hits = retrieve(driver, "engine failure on approach", model=embed_model, top_k=3)
    assert isinstance(hits, list)
    assert len(hits) <= 3


def test_retrieve_hit_has_required_keys(driver, embed_model):
    from src.rag import retrieve
    hits = retrieve(driver, "spatial disorientation at night", model=embed_model, top_k=1)
    if not hits:
        pytest.skip("No incidents in the index")
    hit = hits[0]
    assert "acn" in hit
    assert "score" in hit
    assert 0.0 <= hit["score"] <= 1.0


def test_retrieve_respects_top_k(driver, embed_model):
    from src.rag import retrieve
    hits = retrieve(driver, "tail rotor failure", model=embed_model, top_k=5)
    assert len(hits) <= 5


def test_enrich_returns_dict(driver, embed_model):
    from src.rag import enrich, retrieve
    hits = retrieve(driver, "VFR into IMC", model=embed_model, top_k=2)
    if not hits:
        pytest.skip("No incidents in the index")
    acns = [h["acn"] for h in hits]
    result = enrich(driver, acns)
    assert isinstance(result, dict)
    for acn in acns:
        if acn in result:
            assert "mission" in result[acn]
            assert "anomalies" in result[acn]
            assert isinstance(result[acn]["anomalies"], list)


def test_enrich_unknown_acn_ignored(driver):
    from src.rag import enrich
    result = enrich(driver, ["ACN_THAT_DOES_NOT_EXIST"])
    assert result == {}
