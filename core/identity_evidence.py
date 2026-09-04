"""
core/identity_evidence.py
Per-track identity evidence accumulation, multi-frame voting, and locking.

A track never shows a name after one match. It walks through states as evidence
accumulates:

    UNKNOWN  -> no accepted observations yet
    CANDIDATE-> some accepted votes, not enough to confirm
    CONFIRMED-> enough votes + winner ratio + >=1 strong observation
    LOCKED   -> strong, stable identity; stop re-recognising to save CPU
    AMBIGUOUS-> best vs second-best too close, repeatedly
    REVIEW_REQUIRED -> conflicting confirmed evidence

Pure logic, no models -> unit tested.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from config import settings

UNKNOWN = "UNKNOWN"
CANDIDATE = "CANDIDATE"
CONFIRMED = "CONFIRMED"
LOCKED = "LOCKED"
AMBIGUOUS = "AMBIGUOUS"
REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class Observation:
    employee_id: Optional[str]
    similarity: float
    quality: float
    strong: bool
    accepted: bool
    ambiguous: bool
    frame_ts: float = 0.0


@dataclass
class TrackIdentity:
    track_id: int
    status: str = UNKNOWN
    locked_id: Optional[str] = None
    votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    strong_votes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    body_votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    n_accepted: int = 0
    n_ambiguous: int = 0
    n_body: int = 0
    sims: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    best_sim: float = 0.0
    history: Deque[Observation] = field(default_factory=lambda: deque(maxlen=64))

    # ── read-outs ─────────────────────────────────────────────────────────
    @property
    def identity(self) -> Optional[str]:
        if self.locked_id:
            return self.locked_id
        if not self.votes:
            return None
        return max(self.votes, key=self.votes.get)

    @property
    def mean_sim(self) -> float:
        return sum(self.sims) / len(self.sims) if self.sims else 0.0

    @property
    def vote_ratio(self) -> float:
        if not self.votes:
            return 0.0
        total = sum(self.votes.values())
        return max(self.votes.values()) / total if total else 0.0

    def confidence(self) -> float:
        """0..1 confidence for reporting. Blends vote ratio, mean similarity and
        evidence volume — never a fabricated number."""
        if self.identity is None:
            return 0.0
        vol = min(self.n_accepted / max(settings.IDENTITY_LOCK_THRESHOLD, 1), 1.0)
        sim = min(max(self.mean_sim, 0.0) / 0.7, 1.0)
        base = 0.45 * self.vote_ratio + 0.35 * sim + 0.20 * vol
        if self.status == LOCKED:
            base = min(1.0, base + 0.05)
        return round(float(base), 4)

    def needs_recognition(self) -> bool:
        """Once LOCKED we stop paying for recognition on this track."""
        return self.status != LOCKED


def update(track: TrackIdentity, obs: Observation) -> TrackIdentity:
    """Fold one observation into a track's evidence and recompute its status."""
    track.history.append(obs)
    if obs.similarity > 0:
        track.sims.append(obs.similarity)
        track.best_sim = max(track.best_sim, obs.similarity)

    if obs.ambiguous:
        track.n_ambiguous += 1

    if obs.accepted and obs.employee_id is not None:
        # weight the vote by quality: strong obs count more, weak obs count a little
        weight = 1.0 if obs.strong else 0.4
        track.votes[obs.employee_id] += weight
        if obs.strong:
            if obs.similarity >= settings.FACE_SIM_STRONG:
                # only confirmation-grade similarity may count as a
                # STRONG vote: borderline (0.38-0.50) lookalikes can
                # vote but can never satisfy the confirm requirement
                track.strong_votes[obs.employee_id] += 1
        track.n_accepted += 1

    _recompute_status(track)
    return track


def _recompute_status(t: TrackIdentity) -> None:
    if t.status == LOCKED:
        _maybe_unlock(t)
        return

    winner = t.identity
    if winner is None:
        t.status = AMBIGUOUS if t.n_ambiguous >= settings.MIN_IDENTITY_VOTES else UNKNOWN
        return

    strong_for_winner = t.strong_votes.get(winner, 0)
    ratio = t.vote_ratio

    # conflicting confirmed evidence -> review
    strong_ids = [e for e, c in t.strong_votes.items() if c >= settings.MIN_STRONG_VOTES]
    if len(strong_ids) >= 2:
        t.status = REVIEW_REQUIRED
        return

    if (t.n_accepted >= settings.IDENTITY_LOCK_THRESHOLD
            and ratio >= settings.IDENTITY_VOTE_RATIO
            and strong_for_winner >= settings.MIN_STRONG_VOTES
            and t.mean_sim >= settings.IDENTITY_LOCK_MEAN_SIM):
        t.status = LOCKED
        t.locked_id = winner
        return

    if (t.n_accepted >= settings.MIN_IDENTITY_VOTES
            and ratio >= settings.IDENTITY_VOTE_RATIO
            and strong_for_winner >= settings.MIN_STRONG_VOTES):
        t.status = CONFIRMED
        return

    if t.n_accepted >= 1:
        t.status = CANDIDATE
        return

    t.status = UNKNOWN


def _maybe_unlock(t: TrackIdentity) -> None:
    """A locked identity only flips if a DIFFERENT identity accumulates strong
    evidence that beats the locked one by LOCK_OVERRIDE_MARGIN."""
    if not t.votes:
        return
    ranked = sorted(t.votes.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_v = ranked[0]
    if top_id != t.locked_id and top_v - t.votes.get(t.locked_id, 0.0) > 0:
        locked_v = t.votes.get(t.locked_id, 0.0)
        if (top_v - locked_v) / max(top_v, 1e-6) >= settings.LOCK_OVERRIDE_MARGIN \
                and t.strong_votes.get(top_id, 0) >= settings.MIN_STRONG_VOTES:
            t.status = REVIEW_REQUIRED   # don't silently switch; flag it
            t.locked_id = None


def recompute(track: TrackIdentity) -> TrackIdentity:
    """Public re-evaluation hook (used after evidence inheritance on re-linked
    fragments) so inherited votes are reflected in the track status."""
    _recompute_status(track)
    return track


def inherit(dst: TrackIdentity, donor: TrackIdentity) -> TrackIdentity:
    """Carry evidence from a lost track to its appearance-linked continuation —
    as a HEAD START ONLY. Clothing/appearance similarity is not a biometric,
    so inherited evidence must NEVER be able to confirm an identity by itself:
      - votes are damped (x0.5)
      - strong votes are NOT inherited (confirmation demands FRESH strong
        face observations on the new track)
      - accepted count is capped 2 below the confirmation threshold
    A wrongly-linked stranger therefore stays CANDIDATE until their own face
    disproves or proves the identity."""
    for emp, v in donor.votes.items():
        dst.votes[emp] += 0.5 * v
    dst.n_accepted += min(donor.n_accepted,
                          max(settings.MIN_IDENTITY_VOTES - 2, 0))
    for s in list(donor.sims)[-8:]:
        dst.sims.append(s)
    _recompute_status(dst)
    return dst


def support(t: TrackIdentity, employee_id: str, similarity: float) -> TrackIdentity:
    """Body-ReID SUPPORTING evidence. Adds a small fractional vote, capped per
    track+employee, and touches neither n_accepted, strong votes, nor face
    similarities — so body evidence can NEVER confirm an identity by itself
    and can never swamp face votes ("clothing is not a biometric")."""
    if t.body_votes[employee_id] >= settings.BODY_SUPPORT_CAP:
        return t
    add = min(settings.BODY_SUPPORT_VOTE,
              settings.BODY_SUPPORT_CAP - t.body_votes[employee_id])
    t.body_votes[employee_id] += add
    t.votes[employee_id] += add
    t.n_body += 1
    _recompute_status(t)
    return t


def weaken(t: TrackIdentity, employee_id: str) -> TrackIdentity:
    """Strip a track's claim to an identity (same-person-in-two-places
    conflict): the same employee cannot occupy two visible locations at the
    same timestamp. The demoted track keeps a small residual vote and must
    re-prove the identity with fresh face evidence."""
    t.votes[employee_id] *= 0.25
    t.strong_votes[employee_id] = 0
    t.n_accepted = min(t.n_accepted, max(settings.MIN_IDENTITY_VOTES - 2, 0))
    t.locked_id = None
    if t.status in (CONFIRMED, LOCKED):
        t.status = CANDIDATE
    _recompute_status(t)
    return t
