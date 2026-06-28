# Helicopter Incident Analysis — Exploratory Statistical Report

**Data:** NASA ASRS voluntary incident reports, helicopter subset, Jan 1988 – May 2026  
**Sample:** 2,557 incidents  
**Analysis:** `src/analysis.py`

---

## 1. Data Overview

### Incident volume by flight conditions

| Condition | Count | % |
|---|---|---|
| VMC (clear) | 1,865 | 72.9% |
| IMC (instrument conditions) | 122 | 4.8% |
| Marginal VMC | 97 | 3.8% |
| Mixed (conditions changed in flight) | 80 | 3.1% |
| Not recorded | 393 | 15.4% |

The dataset is heavily skewed toward VMC, which reflects both the reality that most helicopter operations occur in clear weather and the voluntary reporting structure of ASRS.

### Incident volume by light condition

| Light | Count |
|---|---|
| Daylight | 1,793 |
| Night | 340 |
| Dusk | 90 |
| Dawn | 36 |

### Primary problem distribution

Human Factors dominates the dataset (58.7% of incidents), which is consistent with the broader aviation safety literature finding that the majority of accidents have a human error component regardless of conditions.

| Primary problem | Count |
|---|---|
| Human Factors | 1,500 |
| Aircraft | 254 |
| Ambiguous | 171 |
| Procedure | 169 |
| Company Policy | 53 |
| Environment (non-weather) | 45 |
| Airport | 44 |
| ATC Equipment / Facilities | 42 |
| Airspace Structure | 40 |
| Weather | 95 |
| Other | 29 |

### Incident trend over time

Reporting peaked in the late 1980s–mid 1990s and has declined since, likely reflecting both changes in ASRS reporting culture and shifts in helicopter fleet size. The uneven distribution across years should be kept in mind when interpreting findings.

### Visibility distribution

Visibility was recorded for 1,318 of 2,557 incidents (51.5%). Values range from 0 to 9,999 statute miles — the 9,999 value is a data artifact representing "unlimited" visibility. The median is 10 SM and the mean is inflated to 24.3 SM by these outlier codes. All visibility analyses use raw reported values and are best interpreted through medians rather than means.

---

## 2. Hypothesis Tests

### H1 — Do flight conditions predict the type of primary problem?

**Test:** Chi-square test of independence  
**Variables:** flight conditions (VMC / IMC / Marginal) × primary problem (top 5 categories + Other)  
**Excluded:** Mixed conditions (ambiguous by definition)  
**n = 2,084**

| | Aircraft | Ambiguous | Human Factors | Other | Procedure | Weather |
|---|---|---|---|---|---|---|
| **IMC** | 9 | 5 | 63 | 14 | 7 | 24 |
| **Marginal** | 1 | 6 | 61 | 6 | 2 | 21 |
| **VMC** | 206 | 142 | 1,177 | 242 | 78 | 20 |

**Result:** χ²(10) = 258.84, p < 0.001, Cramér's V = 0.249

**Interpretation:** There is a statistically significant, moderately strong association between flight conditions and primary problem type. The most striking difference is in the Weather category: 19.7% of IMC incidents and 21.6% of Marginal incidents list Weather as the primary problem, compared to only 1.1% of VMC incidents. Conversely, Human Factors dominates across all conditions (VMC: 63.1%, IMC: 51.6%, Marginal: 62.9%), suggesting that pilot decision-making is implicated in the majority of incidents regardless of weather.

**Note:** Two cells have expected counts below 5 (the chi-square reliability threshold), specifically in the Marginal row. Results for the Marginal group should be interpreted with caution and treated as indicative rather than conclusive.

---

### H2 — Do EMS missions fly into worse conditions than other missions?

**Test:** Chi-square test of independence  
**Variables:** mission type (EMS / Non-EMS) × flight conditions (all four categories)  
**n = 2,164**

| | IMC | Marginal | Mixed | VMC |
|---|---|---|---|---|
| **EMS** | 29 | 26 | 25 | 271 |
| **Non-EMS** | 93 | 71 | 55 | 1,594 |
| **EMS %** | 8.3% | 7.4% | 7.1% | 77.2% |
| **Non-EMS %** | 5.1% | 3.9% | 3.0% | 87.9% |

**Result:** χ²(3) = 30.33, p < 0.001, Cramér's V = 0.118

**Interpretation:** EMS missions are significantly more likely to occur in non-VMC conditions. 22.8% of EMS incidents occur in IMC, Marginal, or Mixed conditions, compared to 12.1% for non-EMS missions — nearly double the rate. This supports the well-documented mission pressure hypothesis: EMS crews face implicit (and sometimes explicit) pressure to launch or continue into deteriorating conditions when a life is at stake. The effect size is weak (V = 0.118), meaning conditions alone do not define EMS risk — but the directional finding is clear and consistent with NTSB helicopter EMS safety studies. All expected cell counts exceed 5, so the test is fully reliable.

---

### H3 — Within VMC incidents, does night produce a different problem profile?

**Test:** Chi-square test of independence  
**Variables:** light condition (Daylight / Night+Low-light) × primary problem  
**Restricted to:** VMC incidents only  
**n = 1,865**

| | Aircraft | Ambiguous | Human Factors | Other | Procedure | Weather |
|---|---|---|---|---|---|---|
| **Daylight** | 164 | 114 | 967 | 180 | 60 | 15 |
| **Night/Low-light** | 42 | 28 | 210 | 62 | 18 | 5 |
| **Daylight %** | 10.9% | 7.6% | 64.5% | 12.0% | 4.0% | 1.0% |
| **Night %** | 11.5% | 7.7% | 57.5% | 17.0% | 4.9% | 1.4% |

**Result:** χ²(5) = 8.93, p = 0.112, Cramér's V = 0.069

**Interpretation:** **No significant association found.** Within VMC conditions, the problem profile at night is not meaningfully different from daytime. Human Factors dominates in both (64.5% vs 57.5%), and no category shows a dramatic shift. This is a null result — the "dark night VMC trap" hypothesis is not supported by this dataset, at least not at the level of primary problem categorisation. It is possible that finer-grained analysis of the anomaly types (e.g., specifically spatial disorientation events, which are subcategories within Inflight Events) would reveal night-specific patterns not visible at this level.

---

## 3. Visibility by Flight Condition

**Test:** Kruskal-Wallis H test (non-parametric, used because visibility is right-skewed)  
**Groups:** VMC / IMC / Marginal  
**n:** VMC=1,055, IMC=85, Marginal=75 (subset with visibility recorded)

| Condition | Median visibility (SM) |
|---|---|
| VMC | 10.0 |
| Marginal | 4.0 |
| IMC | 3.0 |

**Kruskal-Wallis:** H = 253.86, p < 0.001

**Pairwise Mann-Whitney (Bonferroni corrected):**

| Comparison | p (corrected) | Significant? |
|---|---|---|
| VMC vs IMC | < 0.001 | Yes |
| VMC vs Marginal | < 0.001 | Yes |
| IMC vs Marginal | 0.474 | No |

**Interpretation:** Visibility differs significantly across conditions overall. VMC incidents report a median visibility of 10 SM (the standard ceiling for "good" weather). Both IMC and Marginal incidents have substantially lower reported visibility (3–4 SM), and crucially, **IMC and Marginal are not significantly different from each other in reported visibility**. This is the strongest quantitative support for the "danger zone" hypothesis: Marginal conditions are not a mild step below VMC — they are statistically indistinguishable from full IMC in terms of actual visibility on the ground. Pilots who decide to fly VFR in Marginal conditions are doing so in visibility that is effectively the same as IMC.

---

## 4. Logistic Regression — Predictors of Inflight Event Anomalies

**Outcome:** Whether an incident involved an Inflight Event anomaly (e.g. weather encounter, VFR into IMC, loss of control, CFIT)  
**Predictors:** IMC conditions, Marginal conditions, night/low-light, EMS mission  
**Reference categories:** VMC, Daylight, Non-EMS  
**n = 2,557**

| Predictor | Odds Ratio | 95% CI | p-value |
|---|---|---|---|
| IMC conditions | **7.79** | 5.32 – 11.40 | < 0.001 |
| Marginal conditions | **8.13** | 5.32 – 12.42 | < 0.001 |
| Night / low-light | **1.71** | 1.33 – 2.21 | < 0.001 |
| EMS mission | 1.00 | 0.75 – 1.33 | 0.989 |

**Model fit:** Pseudo-R² (McFadden) = 0.088, AIC = 2,229.54

**Interpretation:**

- **IMC and Marginal conditions** are both associated with roughly 8× higher odds of an Inflight Event anomaly compared to VMC, and their confidence intervals overlap substantially. This is the clearest finding in the dataset: Marginal conditions carry the same in-flight risk as full IMC, not a lesser one.

- **Night/low-light** independently increases the odds of an Inflight Event by 1.7× even after controlling for weather conditions and mission type. This suggests that reduced visual reference is a risk factor in its own right, separate from the official weather classification.

- **EMS mission type has no independent effect** (OR ≈ 1.0, p = 0.989). When you control for conditions, flying an EMS mission does not increase the probability of an Inflight Event. The elevated risk for EMS crews identified in H2 appears to come entirely from the fact that they fly in worse conditions — not from something inherent to the mission type itself. In other words, it is the weather they accept, not the mission label, that puts them at risk.

- The low pseudo-R² (0.088) is expected: incident causation is complex and the model uses only four predictors. The model is not intended as a predictive tool but as a way to quantify the independent contribution of each factor.

---

## 5. Summary of Findings

| Finding | Result |
|---|---|
| Flight conditions predict problem type | ✓ Significant (p < 0.001, V = 0.25) |
| EMS missions fly into worse conditions | ✓ Significant (p < 0.001, V = 0.12) |
| Night changes problem profile within VMC | ✗ Not significant (p = 0.112) |
| Marginal and IMC visibility are different | ✗ Not significant — they are the same |
| Marginal/IMC → 8× higher Inflight Event odds | ✓ Significant (p < 0.001) |
| Night → 1.7× higher Inflight Event odds | ✓ Significant (p < 0.001) |
| EMS independently increases Inflight Event odds | ✗ Not significant (OR ≈ 1.0) |

The most actionable finding is the equivalence of Marginal and IMC conditions. A pilot who classifies their flight as "Marginal VMC" and treats it as meaningfully safer than IMC is not supported by this data.

---

## 6. Limitations

- **Voluntary reporting:** ASRS reports are self-submitted. Serious incidents that result in fatalities are underrepresented (the pilot cannot report). Near-misses and procedural incidents are overrepresented. Results describe patterns within what gets reported, not across all helicopter operations.
- **No denominator:** Without flight-hours by condition type, incidence *rates* cannot be computed. A higher count of VMC incidents does not mean VMC is more dangerous than IMC.
- **H1 cell sparsity:** Two cells in the Marginal row of H1 have expected counts below 5. A Fisher's exact test or collapsing of categories would be appropriate for confirmatory work.
- **Visibility completeness:** Visibility was recorded for only 51.5% of incidents. The subset with visibility may not be representative.
- **Visibility coding:** Values of 9,999 SM appear to be a system code for unlimited visibility, not a real measurement. These were not filtered in the Kruskal-Wallis analysis and may inflate group variance.
- **Anomaly multi-labelling:** Each incident can carry multiple anomaly labels. The logistic regression outcome (any Inflight Event) does not distinguish between incidents where Inflight Event was the primary anomaly versus a secondary one.
