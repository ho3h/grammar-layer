"""Linear and manifold-trajectory steering hooks (PRD §8.1)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from neograph.config import SAE
from neograph.util import get_logger

log = get_logger("neograph.steering")


@dataclass
class SteeringSpec:
    """A trajectory steer is a (n_wp, d_model) tensor of waypoint centroids."""

    trajectory: torch.Tensor  # (n_wp, d_model)
    alpha: float = 0.7
    cyclic: bool = False
    label: str = ""


def linear_steer_hook(activations: torch.Tensor, hook, v: torch.Tensor, alpha: float):
    """Add α·v to the last position's residual stream."""
    # activations: (batch, seq, d_model)
    activations[:, -1, :] = activations[:, -1, :] + alpha * v.to(activations.dtype).to(activations.device)
    return activations


def manifold_steer_hook(
    activations: torch.Tensor,
    hook,
    traj: torch.Tensor,
    alpha: float,
    t_step: int,
    cyclic: bool = False,
):
    """Move the residual stream toward waypoint `t_step` of the trajectory."""
    n_wp = traj.shape[0]
    if cyclic:
        idx = t_step % n_wp
    else:
        idx = min(t_step, n_wp - 1)
    target = traj[idx].to(activations.dtype).to(activations.device)
    activations[:, -1, :] = activations[:, -1, :] + alpha * (target - activations[:, -1, :])
    return activations


def attach_linear_steer(
    model,
    v: torch.Tensor,
    alpha: float = 0.7,
    hook_name: str = SAE.hook_name,
) -> Callable[[], None]:
    """Register a hook returning a remove() function."""
    model.add_hook(hook_name, partial(linear_steer_hook, v=v, alpha=alpha))

    def remove() -> None:
        model.reset_hooks()

    return remove


def attach_manifold_steer(
    model,
    spec: SteeringSpec,
    t_step: int,
    hook_name: str = SAE.hook_name,
) -> Callable[[], None]:
    """Register a manifold-trajectory hook at generation step `t_step`."""
    model.add_hook(
        hook_name,
        partial(
            manifold_steer_hook,
            traj=spec.trajectory,
            alpha=spec.alpha,
            t_step=t_step,
            cyclic=spec.cyclic,
        ),
    )

    def remove() -> None:
        model.reset_hooks()

    return remove
