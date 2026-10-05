"""The decision under study, and the sweep that establishes its right answer.

One decision drives everything measured in this stage: **should an
undeterminable parameter be estimated anyway, or fixed at its nominal?**

Project 12 showed that ``theta`` carries about 1/87 of ``vth0``'s sensitivity.
No measurement plan recovers it.  The tempting conclusion -- fix it and stop
wasting the budget -- is right only if the value it is fixed *at* is close
enough to the truth.  Fixed well, it removes a nuisance direction and every
other parameter improves.  Fixed badly, it injects a bias that no measurement
plan can undo.

So the decision depends on a quantity the Fisher matrix does not contain: how
far the nominal can be trusted.  ``breakeven_sweep`` walks that quantity from
2% to 60% and measures which choice wins, giving a *ground truth decision
boundary* to compare decision-makers against.

Measured break-even: the drop is better up to roughly 20% nominal error and
worse beyond it.  That number is the point of the stage.  Nobody writing a
threshold by hand knew it -- the provenance rule in ``objective.py`` guesses
5%, which is wrong by a factor of four, and the sensitivity-only rule never
looks at trust at all and is wrong above the boundary.

A third case was tried and dropped.  The idea was that when the parameters
feed a corner generator, a c-optimal design on ``vth0`` should beat D-optimal
on ``vth0``'s uncertainty.  Measured over 60 noise realisations it won by
4.7%, which is inside the sampling error of a standard deviation estimated
from 60 samples (about 9%).  It is reported as not established rather than
quoted as a win.  Interestingly c-optimal with *all seven* parameters free is
much worse than D (0.50% against 0.24%): precision cannot be bought on one
parameter while its strongly correlated partner floats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

ALL_PARAMS = ["vth0", "ss", "mu0", "theta", "rs", "eta", "vsat"]
WITHOUT_THETA = [n for n in ALL_PARAMS if n != "theta"]

#: Nominal relative errors swept to locate the decision boundary.
TRUST_LEVELS: Sequence[float] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)

#: The parameter whose identifiability is in question.
SUBJECT = "theta"


@dataclass(frozen=True)
class TrustCase:
    """One point on the sweep: how far ``SUBJECT``'s nominal can be trusted."""

    rel_uncertainty: float

    @property
    def key(self) -> str:
        return f"trust_{int(round(self.rel_uncertainty * 100)):02d}pct"

    def provenance(self) -> Dict[str, Dict]:
        prov = {n: {"source": "extracted on this process",
                    "rel_uncertainty": 0.02} for n in ALL_PARAMS}
        prov[SUBJECT] = {
            "source": ("measured on this process" if self.rel_uncertainty <= 0.05
                       else "carried over from another process"
                       if self.rel_uncertainty <= 0.20
                       else "generic default, never measured here"),
            "rel_uncertainty": self.rel_uncertainty}
        return prov

    def purpose(self) -> str:
        return (
            "Predict drain current across the full operating range for "
            "circuit simulation. If a parameter is left out of the objective "
            "it is held fixed at its nominal value during extraction, and the "
            f"nominal for {SUBJECT} is believed to about "
            f"{self.rel_uncertainty * 100:.0f} percent.")

    def believed(self, truth: Dict[str, float]) -> Dict[str, float]:
        """What we think the parameters are: one sigma off truth for SUBJECT.

        The design is built at the believed values, not at the truth, because
        that is the situation a real extraction is in.  Only the evaluation
        sees the truth.
        """
        bel = dict(truth)
        bel[SUBJECT] = truth[SUBJECT] * (1.0 + self.rel_uncertainty)
        return bel


def breakeven_sweep(levels: Sequence[float] = TRUST_LEVELS) -> List[TrustCase]:
    return [TrustCase(float(x)) for x in levels]


@dataclass
class DecisionBoundary:
    """Where a decision-maker switches from dropping to keeping ``SUBJECT``."""

    name: str
    drops_at: Dict[float, bool] = field(default_factory=dict)

    def boundary(self) -> float | None:
        """Largest trust level still dropped, or None if it never drops."""
        dropped = [t for t, d in self.drops_at.items() if d]
        return max(dropped) if dropped else None

    def disagreements(self, truth_drops: Dict[float, bool]) -> List[float]:
        return sorted(t for t, d in self.drops_at.items()
                      if t in truth_drops and d != truth_drops[t])

    def summary(self, truth_drops: Dict[float, bool]) -> Dict:
        wrong = self.disagreements(truth_drops)
        return {"name": self.name,
                "decides_drop_up_to": self.boundary(),
                "wrong_at": wrong,
                "n_wrong": len(wrong),
                "n_cases": len(self.drops_at)}
