# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN knowledge distillation

"""Knowledge distillation for SNNs with temporal spike alignment.

No SNN library ships distillation utilities. Every paper has its own script.

Reference: CVPR 2025 — temporal separation + entropy regularization
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


class TemporalDistillationLoss:
    """Temporal-aware distillation loss for SNN→SNN or ANN→SNN transfer.

    Matches per-timestep output distributions from teacher to student,
    with entropy regularization to prevent learning erroneous knowledge.

    Parameters
    ----------
    temperature : float
        Softmax temperature for logit matching.
    alpha : float
        Weight of distillation loss vs task loss (0=task only, 1=distill only).
    entropy_weight : float
        Entropy regularization strength.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5, entropy_weight: float = 0.1):
        self.temperature = temperature
        self.alpha = alpha
        self.entropy_weight = entropy_weight

    def compute(
        self,
        student_logits: np.ndarray[Any, Any],
        teacher_logits: np.ndarray[Any, Any],
        targets: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, float]:
        """Compute distillation loss.

        Parameters
        ----------
        student_logits : ndarray of shape (T, N_classes) or (N_classes,)
            Student output per timestep.
        teacher_logits : ndarray of shape (T, N_classes) or (N_classes,)
        targets : ndarray of shape (N_classes,), optional
            Ground truth for task loss.

        Returns
        -------
        dict with 'total_loss', 'distill_loss', 'task_loss', 'entropy_loss'
        """
        # Soften logits
        s_soft = self._softmax(student_logits / self.temperature)
        t_soft = self._softmax(teacher_logits / self.temperature)

        # KL divergence: sum(t * log(t/s))
        kl = np.sum(t_soft * np.log(np.clip(t_soft / np.clip(s_soft, 1e-10, None), 1e-10, None)))
        distill_loss = float(kl * self.temperature**2)

        # Entropy regularization
        entropy = -float(np.sum(s_soft * np.log(np.clip(s_soft, 1e-10, None))))
        entropy_loss = -self.entropy_weight * entropy

        # Task loss (cross-entropy with targets)
        task_loss = 0.0
        if targets is not None:
            s_logits = student_logits if student_logits.ndim == 1 else student_logits.mean(axis=0)
            s_prob = self._softmax(s_logits)
            task_loss = -float(np.sum(targets * np.log(np.clip(s_prob, 1e-10, None))))

        total = self.alpha * distill_loss + (1 - self.alpha) * task_loss + entropy_loss

        return {
            "total_loss": total,
            "distill_loss": distill_loss,
            "task_loss": task_loss,
            "entropy_loss": entropy_loss,
        }

    @staticmethod
    def _softmax(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if x.ndim > 1:
            x = x.mean(axis=0)
        e = np.exp(x - x.max())
        probs: np.ndarray[Any, Any] = e / e.sum()
        return probs


@dataclass
class SelfDistiller:
    """Self-distillation: use extended-T model as implicit teacher.

    Run the same model at T_teacher timesteps (more accurate) to
    generate soft targets for training at T_student timesteps (faster).

    Parameters
    ----------
    T_teacher : int
        Timesteps for teacher forward pass.
    T_student : int
        Timesteps for student forward pass.
    temperature : float
    """

    T_teacher: int = 32
    T_student: int = 8
    temperature: float = 3.0

    def generate_targets(
        self,
        run_fn: Callable[[np.ndarray[Any, Any], int], np.ndarray[Any, Any]],
        inputs: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Run model at T_teacher steps to generate soft targets.

        Parameters
        ----------
        run_fn : callable(inputs, T) -> logits
        inputs : ndarray

        Returns
        -------
        ndarray — soft targets from teacher
        """
        teacher_logits = run_fn(inputs, self.T_teacher)
        return self._softmax(teacher_logits / self.temperature)

    @staticmethod
    def _softmax(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        e = np.exp(x - x.max())
        probs: np.ndarray[Any, Any] = e / e.sum()
        return probs
