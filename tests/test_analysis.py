import pytest

from src.analysis import (
    _cramers_v,
    chi_conditions_vs_problem,
    chi_ems_vs_conditions,
    chi_night_vs_problem,
    descriptive,
    kruskal_visibility_by_condition,
    logistic_inflight_event,
)
from src.ingest import RAW_CSV, clean, load


@pytest.fixture(scope="module")
def df():
    if not RAW_CSV.exists():
        pytest.skip("Raw data file not available")
    return clean(load())


# --- unit test: helper ---

def test_cramers_v_perfect_association():
    # 2x2 table where chi2 == n gives V == 1
    assert _cramers_v(chi2=100, n=100, r=2, c=2) == pytest.approx(1.0)


def test_cramers_v_no_association():
    assert _cramers_v(chi2=0, n=100, r=2, c=2) == pytest.approx(0.0)


# --- descriptive ---

def test_descriptive_keys(df):
    result = descriptive(df)
    assert set(result) == {
        "n_total", "flight_conditions", "light", "mission",
        "primary_problem", "incidents_by_year", "visibility_summary",
    }


def test_descriptive_total_matches(df):
    assert descriptive(df)["n_total"] == len(df)


# --- chi-square tests: shared shape checks ---

@pytest.mark.parametrize("fn", [
    chi_conditions_vs_problem,
    chi_ems_vs_conditions,
    chi_night_vs_problem,
])
def test_chi_result_keys(df, fn):
    result = fn(df)
    assert {"chi2", "p_value", "dof", "cramers_v", "n", "min_expected_count", "table"} <= set(result)


@pytest.mark.parametrize("fn", [
    chi_conditions_vs_problem,
    chi_ems_vs_conditions,
    chi_night_vs_problem,
])
def test_chi_p_value_in_range(df, fn):
    p = fn(df)["p_value"]
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("fn", [
    chi_conditions_vs_problem,
    chi_ems_vs_conditions,
    chi_night_vs_problem,
])
def test_cramers_v_in_range(df, fn):
    v = fn(df)["cramers_v"]
    assert 0.0 <= v <= 1.0


# --- visibility test ---

def test_visibility_result_keys(df):
    result = kruskal_visibility_by_condition(df)
    assert {"kruskal_h", "kruskal_p", "group_medians", "group_ns", "pairwise"} <= set(result)


def test_visibility_p_in_range(df):
    assert 0.0 <= kruskal_visibility_by_condition(df)["kruskal_p"] <= 1.0


def test_visibility_pairwise_has_all_pairs(df):
    result = kruskal_visibility_by_condition(df)
    assert len(result["pairwise"]) == 3  # VMC/IMC/Marginal → 3 pairs


# --- logistic regression ---

def test_logistic_result_keys(df):
    result = logistic_inflight_event(df)
    assert {"n", "pseudo_r2", "aic", "odds_ratios", "p_values", "ci_lower", "ci_upper"} <= set(result)


def test_logistic_predictors_present(df):
    result = logistic_inflight_event(df)
    for key in ("is_imc", "is_marginal", "is_night", "is_ems"):
        assert key in result["odds_ratios"]


def test_logistic_odds_ratios_positive(df):
    result = logistic_inflight_event(df)
    assert all(v > 0 for v in result["odds_ratios"].values())
