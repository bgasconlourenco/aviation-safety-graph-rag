"""Phase 4: exploratory hypothesis tests over cleaned ASRS data."""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_TOP_PROBLEMS = ["Human Factors", "Procedure", "Weather", "Aircraft", "Ambiguous"]
_MAIN_CONDITIONS = ["VMC", "IMC", "Marginal"]


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Effect size for chi-square. 0 = no association, 1 = perfect association."""
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def _chi_result(table: pd.DataFrame) -> dict:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    v = _cramers_v(chi2, n, *table.shape)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 4),
        "dof": dof,
        "cramers_v": round(v, 4),
        "n": n,
        "min_expected_count": round(expected.min(), 2),
        "table": table,
    }


def descriptive(df: pd.DataFrame) -> dict:
    """Summary counts and distributions across key variables."""
    return {
        "n_total": len(df),
        "flight_conditions": df["flight_conditions"].value_counts().to_dict(),
        "light": df["light"].value_counts().to_dict(),
        "mission": df["mission"].value_counts().to_dict(),
        "primary_problem": df["primary_problem"].value_counts().to_dict(),
        "incidents_by_year": df.groupby(df["date"].dt.year).size().to_dict(),
        "visibility_summary": df["visibility_sm"].describe().round(2).to_dict(),
    }


def chi_conditions_vs_problem(df: pd.DataFrame) -> dict:
    """H1: Do different flight conditions produce different primary problems?

    Chi-square test of independence: flight_conditions × primary_problem.
    Mixed excluded (ambiguous by definition). Tail problems collapsed to Other.
    """
    sub = df[df["flight_conditions"].isin(_MAIN_CONDITIONS)].copy()
    sub["problem_grouped"] = sub["primary_problem"].where(
        sub["primary_problem"].isin(_TOP_PROBLEMS), other="Other"
    )
    table = pd.crosstab(sub["flight_conditions"], sub["problem_grouped"])
    return _chi_result(table)


def chi_ems_vs_conditions(df: pd.DataFrame) -> dict:
    """H2: Do EMS missions fly into worse conditions than other missions?

    Chi-square test of independence: mission_type (EMS / Non-EMS) × flight_conditions.
    """
    sub = df.copy()
    sub["mission_type"] = sub["mission"].apply(
        lambda m: "EMS" if m in ("Ambulance", "Other EMS") else "Non-EMS"
    )
    table = pd.crosstab(sub["mission_type"], sub["flight_conditions"])
    return _chi_result(table)


def chi_night_vs_problem(df: pd.DataFrame) -> dict:
    """H3: Within VMC, does night produce a different problem profile than daylight?

    Chi-square test of independence restricted to VMC incidents:
    light_grouped (Daylight / Night+Low-light) × primary_problem.
    """
    vmc = df[df["flight_conditions"] == "VMC"].copy()
    vmc["light_grouped"] = vmc["light"].apply(
        lambda light: "Daylight" if light == "Daylight" else "Night/Low-light"
    )
    vmc["problem_grouped"] = vmc["primary_problem"].where(
        vmc["primary_problem"].isin(_TOP_PROBLEMS), other="Other"
    )
    table = pd.crosstab(vmc["light_grouped"], vmc["problem_grouped"])
    return _chi_result(table)


def kruskal_visibility_by_condition(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis + pairwise Mann-Whitney on visibility_sm across flight conditions.

    Non-parametric because visibility is right-skewed and not normally distributed.
    Bonferroni correction applied to pairwise comparisons.
    """
    sub = df[df["flight_conditions"].isin(_MAIN_CONDITIONS)].dropna(subset=["visibility_sm"])
    groups = {
        cond: grp["visibility_sm"].values
        for cond, grp in sub.groupby("flight_conditions")
    }

    h_stat, p_kw = stats.kruskal(*groups.values())

    pairs = list(combinations(groups.keys(), 2))
    n_pairs = len(pairs)
    pairwise = {}
    for a, b in pairs:
        u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        pairwise[f"{a} vs {b}"] = {
            "u_stat": round(float(u), 2),
            "p_value": round(float(p), 4),
            "p_bonferroni": round(min(float(p) * n_pairs, 1.0), 4),
        }

    return {
        "kruskal_h": round(float(h_stat), 4),
        "kruskal_p": round(float(p_kw), 4),
        "group_medians": {k: round(float(np.median(v)), 2) for k, v in groups.items()},
        "group_ns": {k: len(v) for k, v in groups.items()},
        "pairwise": pairwise,
    }


def logistic_inflight_event(df: pd.DataFrame) -> dict:
    """Logistic regression: Inflight Event ~ conditions + light + mission type.

    Predicts whether an incident involves an Inflight Event anomaly.
    Reference categories: VMC (conditions), Daylight (light), Non-EMS (mission).
    Returns odds ratios and 95% confidence intervals.
    """
    sub = df.copy()
    sub["outcome"] = sub["anomaly_broad"].apply(
        lambda x: 1 if isinstance(x, list) and "Inflight Event" in x else 0
    )
    sub["is_imc"] = (sub["flight_conditions"] == "IMC").astype(int)
    sub["is_marginal"] = (sub["flight_conditions"] == "Marginal").astype(int)
    sub["is_night"] = sub["light"].isin(["Night", "Dawn", "Dusk"]).astype(int)
    sub["is_ems"] = sub["mission"].isin(["Ambulance", "Other EMS"]).astype(int)

    sub = sub.dropna(subset=["outcome", "is_imc", "is_marginal", "is_night", "is_ems"])

    model = smf.logit(
        "outcome ~ is_imc + is_marginal + is_night + is_ems", data=sub
    ).fit(disp=False)

    conf = np.exp(model.conf_int())

    return {
        "n": int(model.nobs),
        "pseudo_r2": round(float(model.prsquared), 4),
        "aic": round(float(model.aic), 2),
        "odds_ratios": np.exp(model.params).round(4).to_dict(),
        "p_values": model.pvalues.round(4).to_dict(),
        "ci_lower": conf[0].round(4).to_dict(),
        "ci_upper": conf[1].round(4).to_dict(),
    }
