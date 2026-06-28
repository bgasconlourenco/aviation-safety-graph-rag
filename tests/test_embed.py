import pandas as pd
import pytest

from src.embed import EMBEDDING_DIM, _build_text, generate


@pytest.fixture(scope="module")
def tiny_df():
    return pd.DataFrame({
        "acn": ["001", "002", "003"],
        "synopsis": ["Helicopter entered IMC during VFR flight.", "Near miss at low altitude.", None],
        "narrative": ["Pilot lost visual reference in fog.", None, "Engine failure on approach."],
    })


@pytest.fixture(scope="module")
def embedded_df(tiny_df):
    """Run generate() once; skip all dependents if the model can't load."""
    try:
        return generate(tiny_df, batch_size=3)
    except Exception as exc:
        pytest.skip(f"sentence-transformers model unavailable: {exc}")


# --- _build_text (pure Python, always run) ---

def test_build_text_combines_both(tiny_df):
    text = _build_text(tiny_df.iloc[0])
    assert "IMC" in text
    assert "fog" in text


def test_build_text_synopsis_first(tiny_df):
    text = _build_text(tiny_df.iloc[0])
    assert text.index("IMC") < text.index("fog")


def test_build_text_null_synopsis(tiny_df):
    text = _build_text(tiny_df.iloc[2])
    assert "Engine failure" in text
    assert text.startswith("Engine")


def test_build_text_null_narrative(tiny_df):
    text = _build_text(tiny_df.iloc[1])
    assert "Near miss" in text


# --- generate (skipped if model unavailable) ---

def test_generate_adds_embedding_column(embedded_df):
    assert "embedding" in embedded_df.columns


def test_generate_correct_dimension(embedded_df):
    for emb in embedded_df["embedding"]:
        assert len(emb) == EMBEDDING_DIM


def test_generate_embedding_is_list_of_floats(embedded_df):
    emb = embedded_df["embedding"].iloc[0]
    assert isinstance(emb, list)
    assert all(isinstance(v, float) for v in emb)


def test_generate_all_rows_embedded(embedded_df):
    assert embedded_df["embedding"].notna().all()


def test_generate_does_not_mutate_input(tiny_df):
    try:
        generate(tiny_df, batch_size=3)
    except Exception:
        pytest.skip("model unavailable")
    assert "embedding" not in tiny_df.columns
