# Cross-routing functional test — summary

Rosetta corpus: 123 prompts mixing copula-heavy and copula-free text.
For each model we capture per-token SAE activations and rank features by
`copula_mean - noncopula_mean` (specificity for copula tokens).

## Top-N copula features per model (by raw specificity)

### gemma
  n_copula_tokens=72  n_noncopula_tokens=1143
  - feat **15596**  copula_mean=+58.439  noncopula_mean=+0.528  specificity=+57.911
  - feat **13414**  copula_mean=+38.400  noncopula_mean=+0.000  specificity=+38.400
  - feat **10142**  copula_mean=+28.336  noncopula_mean=+5.906  specificity=+22.431
  - feat **13346**  copula_mean=+14.598  noncopula_mean=+0.247  specificity=+14.351
  - feat **13429**  copula_mean=+13.922  noncopula_mean=+0.102  specificity=+13.820
  - feat **15149**  copula_mean=+11.027  noncopula_mean=+0.010  specificity=+11.017
  - feat **2429**  copula_mean=+10.302  noncopula_mean=+0.191  specificity=+10.112
  - feat **12564**  copula_mean=+9.742  noncopula_mean=+0.015  specificity=+9.727
  - feat **15332**  copula_mean=+9.708  noncopula_mean=+0.069  specificity=+9.639
  - feat **2523**  copula_mean=+10.655  noncopula_mean=+1.283  specificity=+9.371

### gpt2
  n_copula_tokens=73  n_noncopula_tokens=1200
  - feat **21496**  copula_mean=+4.623  noncopula_mean=+0.000  specificity=+4.623
  - feat **6651**  copula_mean=+2.924  noncopula_mean=+0.001  specificity=+2.923
  - feat **291**  copula_mean=+2.338  noncopula_mean=+0.005  specificity=+2.333
  - feat **19805**  copula_mean=+2.324  noncopula_mean=+0.000  specificity=+2.324
  - feat **2189**  copula_mean=+2.214  noncopula_mean=+0.001  specificity=+2.214
  - feat **10903**  copula_mean=+2.239  noncopula_mean=+0.054  specificity=+2.185
  - feat **21000**  copula_mean=+2.160  noncopula_mean=+0.010  specificity=+2.150
  - feat **6863**  copula_mean=+3.470  noncopula_mean=+1.678  specificity=+1.792
  - feat **21923**  copula_mean=+1.788  noncopula_mean=+0.011  specificity=+1.777
  - feat **1960**  copula_mean=+1.767  noncopula_mean=+0.110  specificity=+1.657

### pythia_70m
  n_copula_tokens=73  n_noncopula_tokens=1219
  - feat **11271**  copula_mean=+1.950  noncopula_mean=+0.032  specificity=+1.918
  - feat **11255**  copula_mean=+0.806  noncopula_mean=+0.000  specificity=+0.806
  - feat **27730**  copula_mean=+0.760  noncopula_mean=+0.006  specificity=+0.754
  - feat **3308**  copula_mean=+0.687  noncopula_mean=+0.001  specificity=+0.686
  - feat **23527**  copula_mean=+0.647  noncopula_mean=+0.001  specificity=+0.647
  - feat **28810**  copula_mean=+0.573  noncopula_mean=+0.007  specificity=+0.566
  - feat **16511**  copula_mean=+0.600  noncopula_mean=+0.053  specificity=+0.547
  - feat **23666**  copula_mean=+1.531  noncopula_mean=+1.010  specificity=+0.522
  - feat **12173**  copula_mean=+1.376  noncopula_mean=+0.912  specificity=+0.464
  - feat **3617**  copula_mean=+0.449  noncopula_mean=+0.003  specificity=+0.446

### gemma_1_2b
  n_copula_tokens=72  n_noncopula_tokens=1143
  - feat **10323**  copula_mean=+2.774  noncopula_mean=+1.336  specificity=+1.439
  - feat **5943**  copula_mean=+1.387  noncopula_mean=+0.000  specificity=+1.387
  - feat **11942**  copula_mean=+1.275  noncopula_mean=+0.009  specificity=+1.265
  - feat **5162**  copula_mean=+1.152  noncopula_mean=+0.001  specificity=+1.152
  - feat **14100**  copula_mean=+0.871  noncopula_mean=+0.004  specificity=+0.867
  - feat **3555**  copula_mean=+0.799  noncopula_mean=+0.004  specificity=+0.795
  - feat **6578**  copula_mean=+0.754  noncopula_mean=+0.000  specificity=+0.754
  - feat **1023**  copula_mean=+0.659  noncopula_mean=+0.036  specificity=+0.623
  - feat **5541**  copula_mean=+0.592  noncopula_mean=+0.004  specificity=+0.588
  - feat **1873**  copula_mean=+0.555  noncopula_mean=+0.006  specificity=+0.549

## Headline finding

The activation-pattern test rediscovers the named fingerprint features. In Gemma 2 2B, the highest-specificity feature on copula tokens is **f15596** (specificity +57.9, copula_mean = 58.4 vs noncopula_mean = 0.5). The next two are **f13414** ("statements of existence", a known opposer in the capital fingerprint) and **f10142** ("instances of 'is'", the other named fingerprint feature). No labels enter the calculation. The functional and label-based definitions of "copula detector" agree on the load-bearing features in Gemma 2 2B.

In Pythia 70M, the known fingerprint feature **f23527** ("verb 'is' and various forms") appears at rank 5 by raw specificity. In GPT-2 small, the top copula features by raw specificity include **f6651** ("the verb 'is' followed by various types of content") and **f291** ("the verb 'is' at the beginning of sentences"), which are real copula detectors but are *not* recruited as opposers on the capital prompts — consistent with the label-based finding that GPT-2 owns the grammar vocabulary without routing through it.

The raw-specificity ranking is confounded for cross-routing analysis by content features that happen to co-fire with copula tokens because copula-heavy contexts in the Rosetta corpus are semantically richer than copula-free ones. The selectivity-refined analysis is in `cross_routing_selectivity_summary.md`. The headline result here — Gemma's named fingerprint features are independently rediscovered by their activation pattern — does not depend on the selectivity filter.