#!/bin/bash
# Run when reports/load_bearing_pos10_gemma_9b_l31_50.json and
# reports/behavior_metrics_n300.json both exist.
#
# Computes the 9B-L31 grammar enrichment, summarizes the n=300 behavioral comparison,
# and prints summaries that can be inserted into the writeup.

set -u
cd /Users/tedsandtads/Documents/GitHub/graphgeometry/.claude/worktrees/distracted-khorana-a653e7

echo "=========================================="
echo " 9B at L31 (74% depth) grammar enrichment "
echo "=========================================="
uv run python -c "
import json
labels = json.load(open('data/labels_cache_gemma_9b.json'))
import re
GRAM = re.compile(r'\b(is|to[- ]?be|copula|verb|tense|past tense|present tense|conjunction|article|preposition|punctuation|grammar|function word|determiner|pronoun|auxiliary|are|am|was|were|be|been|being)\b', re.I)
def is_gram(lbl): return bool(lbl and GRAM.search(lbl))

data = json.load(open('reports/load_bearing_pos10_gemma_9b_l31_50.json'))

n_sup_g = 0; n_sup_total = 0
n_opp_g = 0; n_opp_total = 0
cap_n_opp_g = 0; cap_n_opp_total = 0
cap_n_sup_g = 0; cap_n_sup_total = 0
for r in data['results']:
    for e in r.get('topk_supporting', [])[:5]:
        n_sup_total += 1
        lab = labels.get(str(e['feature_index']), {}).get('text', '') if isinstance(labels.get(str(e['feature_index'])), dict) else (labels.get(str(e['feature_index'])) or '')
        if is_gram(lab): n_sup_g += 1
    for e in r.get('topk_opposing', [])[:5]:
        n_opp_total += 1
        lab = labels.get(str(e['feature_index']), {}).get('text', '') if isinstance(labels.get(str(e['feature_index'])), dict) else (labels.get(str(e['feature_index'])) or '')
        if is_gram(lab): n_opp_g += 1
    if r.get('category') == 'capital':
        for e in r.get('topk_supporting', [])[:5]:
            cap_n_sup_total += 1
            lab = labels.get(str(e['feature_index']), {}).get('text', '') if isinstance(labels.get(str(e['feature_index'])), dict) else (labels.get(str(e['feature_index'])) or '')
            if is_gram(lab): cap_n_sup_g += 1
        for e in r.get('topk_opposing', [])[:5]:
            cap_n_opp_total += 1
            lab = labels.get(str(e['feature_index']), {}).get('text', '') if isinstance(labels.get(str(e['feature_index'])), dict) else (labels.get(str(e['feature_index'])) or '')
            if is_gram(lab): cap_n_opp_g += 1

print(f'All 52 prompts:')
print(f'  supporting grammar%: {100*n_sup_g/max(n_sup_total,1):.2f}%')
print(f'  opposing grammar%:   {100*n_opp_g/max(n_opp_total,1):.2f}%')
ratio = (n_opp_g/n_opp_total) / (n_sup_g/n_sup_total) if n_sup_g > 0 else float('inf')
print(f'  enrichment ratio:    {ratio:.2f}x')
print()
print(f'Capital prompts (n={data[\"results\"][0].get(\"category\")}, count below):')
print(f'  supporting grammar%: {100*cap_n_sup_g/max(cap_n_sup_total,1):.2f}%')
print(f'  opposing grammar%:   {100*cap_n_opp_g/max(cap_n_opp_total,1):.2f}%')
cap_ratio = (cap_n_opp_g/cap_n_opp_total) / (cap_n_sup_g/cap_n_sup_total) if cap_n_sup_g > 0 else float('inf')
print(f'  enrichment ratio:    {cap_ratio:.2f}x')
print()
print(f'Top opposers across capital prompts:')
from collections import Counter
opp_counter = Counter()
opp_labels = {}
for r in data['results']:
    if r.get('category') == 'capital':
        for e in r.get('topk_opposing', [])[:5]:
            opp_counter[e['feature_index']] += 1
            opp_labels[e['feature_index']] = labels.get(str(e['feature_index']), {}).get('text', '?') if isinstance(labels.get(str(e['feature_index'])), dict) else (labels.get(str(e['feature_index'])) or '?')
for fid, count in opp_counter.most_common(8):
    print(f'  feat {fid:>6} appears in {count}/6 capitals: {opp_labels[fid][:60]}')
"

echo
echo "=========================================="
echo "  Behavioral metrics n=300 summary        "
echo "=========================================="
if [ -f reports/behavior_metrics_n300.json ]; then
  uv run python -c "
import json
data = json.load(open('reports/behavior_metrics_n300.json'))
print(json.dumps({k: v for k, v in data.get('per_model_means', data).items() if not k.startswith('_')}, indent=2))
print()
print('Pairwise t-tests (Gemma 2 2B vs GPT-2):')
ttests = data.get('ttests', {})
for k, v in ttests.items():
    if 'gemma' in k.lower() and 'gpt2' in k.lower():
        print(f'  {k}: t={v.get(\"t\", 0):.3f} p={v.get(\"p\", 0):.4f}')
" 2>/dev/null || echo "(behavior_metrics_n300.json structure inspection)"
fi
