"""No-training hybrid: SmolVLA advisory target selection + safe sequencer.

The VLA remains the high-level visual advisory.  It is queried only when a
target is needed (reset or after release); its predicted XY action direction is
used to rank RGB-detected parcel candidates.  Once a candidate is locked, all
executed actions come from :class:`AxisLockedSequencer`, so the VLA cannot
oscillate, switch targets mid-pick, or violate the phase mask.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .axis_locked_sequencer import AxisLockedSequencer, SequencerTarget
from .rgb_axis_locked_policy import RGBParcelDetector, RGBTarget


@dataclass(frozen=True)
class AdvisoryChoice:
    target: SequencerTarget
    source: RGBTarget
    score: float
    confidence: float
    advice: tuple[float, float, float, float]


@dataclass(frozen=True)
class System2Event:
    primitive: str
    retry: bool
    score: float
    advice: tuple[float, float, float, float]


class VLAAdvisoryHybrid:
    """Use a callable VLA advisor to choose targets, then execute safely."""

    def __init__(
        self,
        advisor: Callable[[np.ndarray, torch.Tensor], torch.Tensor],
        detector: Optional[RGBParcelDetector] = None,
        sequencer: Optional[AxisLockedSequencer] = None,
        fallback_greedy: bool = True,
        min_advisor_norm: float = 0.05,
        max_retries: int = 1,
        retry_threshold: float = 0.15,
        retry_xy_scale: float = 0.006,
    ):
        self.advisor = advisor
        self.detector = detector or RGBParcelDetector()
        self.sequencer = sequencer or AxisLockedSequencer(phase_budget=25)
        self.fallback_greedy = bool(fallback_greedy)
        self.min_advisor_norm = float(min_advisor_norm)
        self.max_retries = int(max_retries)
        self.retry_threshold = float(retry_threshold)
        self.retry_xy_scale = float(retry_xy_scale)
        self.remaining: list[tuple[RGBTarget, SequencerTarget]] = []
        self.current: Optional[AdvisoryChoice] = None
        self.completed: list[AdvisoryChoice] = []
        self.retry_count = 0
        self.events: list[System2Event] = []

    def reset(self, frame_rgb: np.ndarray):
        detected = self.detector.detect(frame_rgb)
        bins = self.detector.detect_bins(frame_rgb)
        bins = {0: 110.0, 1: 16.0, **bins}
        self.remaining = [
            (
                source,
                SequencerTarget(source.xy, self.detector.bin_target(source.color, bins[source.color])),
            )
            for source in detected
        ]
        self.current = None
        self.completed = []
        self.retry_count = 0
        self.events = []
        self.sequencer.reset()

    @staticmethod
    def _action_tuple(advice: torch.Tensor) -> tuple[float, float, float, float]:
        values = advice.detach().float().flatten().cpu().tolist()
        values = (values + [0.0] * 4)[:4]
        return tuple(float(x) for x in values)

    @staticmethod
    def _route_cost(current_xy, target: SequencerTarget) -> float:
        sx, sy = target.parcel_xy
        bx, by = target.bin_xy
        return (
            abs(float(current_xy[0]) - sx)
            + abs(float(current_xy[1]) - sy)
            + abs(sx - bx)
            + abs(sy - by)
        )

    def choose(self, frame_rgb: np.ndarray, state: torch.Tensor) -> AdvisoryChoice:
        if not self.remaining:
            raise RuntimeError("no RGB parcel targets remain")
        advice = self.advisor(frame_rgb, state).detach().float().flatten()
        advice_tuple = self._action_tuple(advice)
        v = advice[:2].cpu().numpy()
        norm = float(np.linalg.norm(v))
        current_xy = state[18:20].detach().float().cpu().numpy()
        # Ranking uses only the VLA's coarse XY intent.  The route-cost term is
        # a tie-breaker/fallback and never changes a high-confidence direction.
        if norm >= self.min_advisor_norm:
            direction = v / norm
            scored = []
            for source, target in self.remaining:
                d = np.asarray(source.xy, dtype=np.float64) - current_xy
                dnorm = float(np.linalg.norm(d))
                cosine = float(np.dot(direction, d / max(dnorm, 1e-8)))
                scored.append((cosine, -self._route_cost(current_xy, target), source, target))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            cosine, _, source, target = scored[0]
            confidence = float((cosine + 1.0) / 2.0)
            choice = AdvisoryChoice(target, source, float(cosine), confidence, advice_tuple)
        else:
            source, target = min(
                self.remaining,
                key=lambda x: self._route_cost(current_xy, x[1]),
            )
            choice = AdvisoryChoice(target, source, 0.0, 0.0, advice_tuple)
        self.remaining.remove((choice.source, choice.target))
        self.current = choice
        self.retry_count = 0
        self.events.append(System2Event("SELECT_TARGET", False, choice.score, advice_tuple))
        self.sequencer.set_target(choice.target)
        return choice

    def _retry_decision(self, frame_rgb: np.ndarray, state: torch.Tensor) -> bool:
        """Decode RETRY/SKIP from a fresh VLA action after a failed grasp.

        Negative gripper/Z means the learned policy still wants to close or
        descend.  Positive gripper/Z is evidence for abandoning the pick.  XY
        magnitude is weak evidence that it still has an active target.
        """
        advice = self.advisor(frame_rgb, state).detach().float().flatten()
        ax, ay, az, grip = self._action_tuple(advice)
        xy = float(np.hypot(ax, ay))
        retry_evidence = max(-grip, -az, 0.5 * xy)
        skip_evidence = max(grip, az, 0.0)
        retry = (
            self.retry_count < self.max_retries
            and retry_evidence >= self.retry_threshold
            and retry_evidence >= skip_evidence
        )
        score = float(retry_evidence - skip_evidence)
        primitive = "RETRY_TARGET" if retry else "NEXT_TARGET"
        self.events.append(System2Event(primitive, retry, score, (ax, ay, az, grip)))
        if retry and self.current is not None:
            # Use the VLA's local XY intention only as a millimetre-scale retry
            # correction.  System 1 still owns the absolute waypoint and all
            # executed actions.
            norm = max(float(np.hypot(ax, ay)), 1e-8)
            dx = self.retry_xy_scale * ax / norm
            dy = self.retry_xy_scale * ay / norm
            old = self.current.target
            corrected = SequencerTarget(
                (old.parcel_xy[0] + dx, old.parcel_xy[1] + dy),
                old.bin_xy,
            )
            self.current = AdvisoryChoice(
                corrected,
                self.current.source,
                self.current.score,
                self.current.confidence,
                self.current.advice,
            )
            self.retry_count += 1
            self.sequencer.reset()
            self.sequencer.set_target(corrected)
        return retry

    def act(self, frame_rgb: np.ndarray, state: torch.Tensor) -> torch.Tensor:
        if self.current is None:
            self.choose(frame_rgb, state)
        action = self.sequencer.act(state)
        if self.sequencer.completed:
            failed = self.sequencer.failed
            if failed and self._retry_decision(frame_rgb, state):
                return action
            self.completed.append(self.current)
            self.current = None
            self.retry_count = 0
            self.sequencer.reset()
        return action

    @property
    def done(self) -> bool:
        return self.current is None and not self.remaining
