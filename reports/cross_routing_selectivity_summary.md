# Cross-routing — selectivity-refined

Reranked the Rosetta-corpus top-N copula features by log10 selectivity
(`log10((copula_mean + ε) / (noncopula_mean + ε))`) and kept only features with
selectivity ≥ 1.0 (i.e., ≥10× more activation on copula than
non-copula tokens) AND copula_mean ≥ 1.0.

## High-selectivity copula features per model

| model | pool size | top feature | log10_sel | label |
|---|---|---|---|---|
| gemma | 15 | f13414 | +4.58 | statements of existence or presence |
| gpt2 | 14 | f21496 | +3.67 | phrases related to uncertainty or questioning |
| pythia_70m | 1 | f11271 | +1.77 |  verbs that signify existence or state |
| gemma_1_2b | 3 | f5943 | +3.14 | the verb "is" in various contexts |

## Cross-routing (high-selectivity copula features only)

| model | capitals | high-sel-copula-in-opposing-top10 (sum) | mean/prompt | pool size |
|---|---|---|---|---|
| gemma | 6 | 15 | 2.50 | 15 |
| gpt2 | 6 | 16 | 2.67 | 14 |
| pythia_70m | 6 | 6 | 1.00 | 1 |
| gemma_1_2b | 6 | 16 | 2.67 | 3 |

**Interpretation:** Selectivity-filtered, the cross-routing picture sharpens. Models
that recruit their *dedicated* copula detectors (high selectivity, exclusive firing
on copula tokens) as opposers on capital prompts are the models with the inversion.

The raw cross_routing_functional analysis was confounded by content features that
happen to co-fire with copula tokens because copula-heavy contexts are semantically
richer. Filtering for log10_selectivity ≥ 1 removes those incidental detectors.