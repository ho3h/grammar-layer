"""P6: Trajectory steering vs linear steering on Gemma 2 2B days-of-week.

Reuses the weekday manifold from P4. If P4 didn't fit a weekday community, we run a
small forced fit on the residual-stream activations at 'Mon'..'Sun' tokens directly.

Reports:
- reports/p6_steering.json — per-direction metrics
- reports/p6_steering.png   — bar chart
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neograph.config import MODEL, PATHS, SAE
from neograph.cypher import NeographClient
from neograph.manifold.fit import fit_community_manifold
from neograph.steering import SteeringSpec, attach_linear_steer, attach_manifold_steer
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.steering.eval")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class StepResult:
    direction: str
    method: str
    target: str
    log_p_target: float
    log_p_day_total: float
    target_hit: bool


def _day_token_ids(tokenizer) -> dict[str, int]:
    out = {}
    for d in DAYS:
        ids = tokenizer.encode(" " + d, add_special_tokens=False)
        out[d] = ids[0] if ids else -1
    return out


def _activation_means(model, hook: str, day_to_tokens: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
    """For each label day, average residual stream at the last token of a small batch of prompts."""
    by_day: dict[str, list[torch.Tensor]] = {d: [] for d, _ in day_to_tokens}
    for day, prompt in day_to_tokens:
        with torch.no_grad():
            tokens = model.to_tokens(prompt, prepend_bos=True)
            _logits, cache = model.run_with_cache(tokens, names_filter=[hook])
            by_day[day].append(cache[hook][0, -1, :].cpu())
    return {d: torch.stack(v).mean(0) for d, v in by_day.items() if v}


def _load_manifold_from_neo4j(c: NeographClient) -> SteeringSpec | None:
    """Find the Leiden community whose AutoInterp labels mention day/week most,
    and use its manifold's waypoints."""
    rows = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE toLower(a.text) CONTAINS 'day' OR toLower(a.text) CONTAINS 'week'
           OR toLower(a.text) CONTAINS 'mon' OR toLower(a.text) CONTAINS 'tues'
        WITH f.communityId AS cid, count(f) AS n
        WHERE cid IS NOT NULL AND n > 5
        RETURN cid, n ORDER BY n DESC LIMIT 1
        """
    )
    if not rows:
        return None
    cid = int(rows[0]["cid"])
    mid_row = c.run(
        "MATCH (m:Manifold) WHERE m.id STARTS WITH 'community-' + toString($cid) + '/' RETURN m.id AS mid, m.is_cyclic AS cyc LIMIT 1",
        cid=cid,
    )
    if not mid_row:
        return None
    mid = mid_row[0]["mid"]
    cyc = bool(mid_row[0].get("cyc", False))
    wps = c.run(
        "MATCH (m:Manifold {id: $mid})-[:HAS_WAYPOINT]->(w:Waypoint) RETURN w.index AS i, w.centroid AS c ORDER BY i",
        mid=mid,
    )
    traj = np.stack([np.asarray(r["c"], dtype=np.float32) for r in wps])
    log.info("Using manifold %s (community %d, %d waypoints, cyclic=%s)", mid, cid, len(wps), cyc)
    return SteeringSpec(trajectory=torch.from_numpy(traj), alpha=0.7, cyclic=cyc, label=mid)


def _forced_weekday_manifold(model) -> SteeringSpec:
    """Build a 7-waypoint cyclic weekday manifold by averaging residual stream
    at the LAST (steering) position of completion-style prompts that elicit day X.

    This places each waypoint in the same region of residual space the steering
    hook will overwrite — avoiding the position-mismatch failure mode."""
    hook = SAE.hook_name
    # Each prompt ends in a position where the next-token target is the named day.
    # The residual at the last token (just before predicting the day) becomes the waypoint.
    templates = [
        "Yesterday was Sunday. Today is",
        "After Sunday comes",
        "Once Sunday passes, the next day is",
    ]
    # Map day → list of prompts that elicit it as next token
    prompts_for: dict[str, list[str]] = {}
    for i, day in enumerate(DAYS):
        prev = DAYS[(i - 1) % len(DAYS)]
        prompts_for[day] = [
            f"Yesterday was {prev}. Today is",
            f"After {prev} comes",
            f"Once {prev} passes, the next day is",
            f"{prev} is over. Now it's",
            f"Today is {prev}. Tomorrow is",  # last-token of "is" — actually elicits day
        ]
    centroids = np.zeros((len(DAYS), 2304), dtype=np.float32)
    for i, day in enumerate(DAYS):
        h_rows = []
        for prompt in prompts_for[day]:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _logits, cache = model.run_with_cache(tokens, names_filter=[hook])
            h_rows.append(cache[hook][0, -1, :].cpu().numpy())  # last position
        centroids[i] = np.mean(h_rows, axis=0)
    return SteeringSpec(
        trajectory=torch.from_numpy(centroids),
        alpha=0.7,
        cyclic=True,
        label="weekday-anchored-lastpos",
    )


def evaluate(model, spec: SteeringSpec, prompts: list[tuple[str, str, str]], hook: str) -> list[StepResult]:
    """For each (direction, prompt, target) run linear and trajectory steers, score."""
    results: list[StepResult] = []
    day_ids = _day_token_ids(model.tokenizer)
    n_wp = spec.trajectory.shape[0]

    for direction, prompt, target in tqdm(prompts, desc="steer-eval"):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        # baseline: no steer
        with torch.no_grad():
            logits_bl = model(tokens)
            probs_bl = logits_bl[0, -1].softmax(-1).cpu()
        # Linear: target − source mean
        try:
            source_day, target_day = direction.split("→")
        except ValueError:
            source_day, target_day = DAYS[0], target
        source_idx = DAYS.index(source_day)
        target_idx = DAYS.index(target_day)
        v = spec.trajectory[target_idx] - spec.trajectory[source_idx]
        # Linear steers ADD α·v (delta vector), so a larger α is reasonable for small v.
        # Manifold steers MOVE α fraction toward target (α ∈ (0, 1]).
        linear_alpha = 4.0
        remove = attach_linear_steer(model, v, alpha=linear_alpha, hook_name=hook)
        with torch.no_grad():
            logits_lin = model(tokens)
            probs_lin = logits_lin[0, -1].softmax(-1).cpu()
        remove()

        # Manifold trajectory: advance step-by-step
        remove = attach_manifold_steer(model, spec, t_step=target_idx, hook_name=hook)
        with torch.no_grad():
            logits_man = model(tokens)
            probs_man = logits_man[0, -1].softmax(-1).cpu()
        remove()

        for method, probs in [("baseline", probs_bl), ("linear", probs_lin), ("manifold", probs_man)]:
            tid = day_ids.get(target_day, -1)
            if tid < 0:
                continue
            lp_target = float(np.log(max(float(probs[tid].item()), 1e-30)))
            day_mass = float(sum(probs[day_ids[d]].item() for d in DAYS if day_ids.get(d, -1) >= 0))
            argmax = int(probs.argmax().item())
            hit = argmax == tid
            results.append(
                StepResult(
                    direction=direction,
                    method=method,
                    target=target_day,
                    log_p_target=lp_target,
                    log_p_day_total=float(np.log(max(day_mass, 1e-30))),
                    target_hit=hit,
                )
            )
    return results


def main() -> int:
    from sae_lens import HookedSAETransformer as HookedTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL.name, device=device)
    model.eval()

    # The PRD steering experiment specifically anchors one waypoint per day.
    # We always build that anchored manifold here; the Neo4j manifold is for inspection.
    log.info("Building day-anchored weekday manifold ...")
    spec = _forced_weekday_manifold(model)
    spec.trajectory = spec.trajectory.to(device)
    log.info("Trajectory shape: %s, alpha=%.2f, cyclic=%s", tuple(spec.trajectory.shape), spec.alpha, spec.cyclic)

    # Prompts that elicit days as top-1 unsteered (verified via debug session)
    directions = [
        ("Monday→Tuesday", "Today is Monday. Tomorrow is", "Tuesday"),
        ("Monday→Wednesday", "Today is Monday. Tomorrow is", "Wednesday"),
        ("Monday→Friday", "Today is Monday. Tomorrow is", "Friday"),
        ("Tuesday→Wednesday", "Today is Tuesday. Tomorrow is", "Wednesday"),
        ("Wednesday→Friday", "Today is Wednesday. Tomorrow is", "Friday"),
        ("Friday→Monday", "Today is Friday. Tomorrow is", "Monday"),
        ("Sunday→Monday", "Today is Sunday. Tomorrow is", "Monday"),
    ]
    results = evaluate(model, spec, directions, SAE.hook_name)

    out_path = PATHS.reports / "p6_steering.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    log.info("Wrote %s (%d rows)", out_path, len(results))

    df = pd.DataFrame([asdict(r) for r in results])
    summary = df.groupby("method").agg(
        {"target_hit": "mean", "log_p_target": "mean", "log_p_day_total": "mean"}
    )
    log.info("Summary:\n%s", summary)
    summary.to_csv(PATHS.reports / "p6_steering_summary.csv")

    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        summary["target_hit"].plot(kind="bar", ax=axs[0], title="Target hit rate")
        (-summary["log_p_day_total"]).plot(kind="bar", ax=axs[1], title="−log P(day) entropy proxy")
        plt.tight_layout()
        plt.savefig(PATHS.reports / "p6_steering.png", dpi=120)
        log.info("Wrote %s", PATHS.reports / "p6_steering.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("Plot generation failed: %s", exc)

    # Soft target: manifold target-hit ≥ 1.5× linear, entropy delta ≥ 0.2 nat
    try:
        lin_hit = float(summary.loc["linear", "target_hit"])
        man_hit = float(summary.loc["manifold", "target_hit"])
        ratio_ok = man_hit >= 1.5 * lin_hit
    except KeyError:
        ratio_ok = False
    exit_marker(
        "steering-eval",
        ok=ratio_ok,
        linear_hit=float(summary.get("target_hit", pd.Series()).get("linear", 0.0)),
        manifold_hit=float(summary.get("target_hit", pd.Series()).get("manifold", 0.0)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
