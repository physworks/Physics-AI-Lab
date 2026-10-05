"""What the instrument can actually resolve, as a function of current.

This module is the reason the whole project exists.  A measurement in deep
subthreshold carries enormous *sensitivity* to the subthreshold swing, so any
design criterion that assumes uniform noise will pile points there.  But that
is exactly where the current approaches the instrument's noise floor and the
*relative* precision collapses.  Sensitivity and information are not the same
thing, and the difference is this function.

    sigma_rel(I) = sqrt( sigma_0^2 + (I_floor / I)^2 )

  * ``sigma_0``   relative noise that survives at large current
  * ``I_floor``   absolute noise floor, dominant as I -> 0

Project 12 measures with both terms but analyses with ``sigma_0`` alone.  That
inconsistency is small for most parameters and not small for ``ss``; see
``outputs/report.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN10 = np.log(10.0)


@dataclass(frozen=True)
class NoiseModel:
    """Instrument noise.  ``floor_aware=False`` reproduces the textbook form."""

    sigma_rel: float = 0.01        # relative noise at large current
    i_floor: float = 1e-12         # absolute noise floor [A]
    floor_aware: bool = True

    def relative(self, current) -> np.ndarray:
        """Relative 1-sigma of a current measurement."""
        i = np.abs(np.asarray(current, dtype=float))
        if not self.floor_aware:
            return np.full(i.shape, self.sigma_rel)
        safe = np.maximum(i, 1e-30)
        return np.sqrt(self.sigma_rel ** 2 + (self.i_floor / safe) ** 2)

    def log10_sigma(self, current) -> np.ndarray:
        """1-sigma of log10(I).

        For small relative error, d log10(I) = dI / (I ln 10), so the log-space
        sigma is the relative sigma divided by ln(10).  The extraction fits
        log10 current, so this is the quantity the Fisher matrix needs.
        """
        return self.relative(current) / LN10

    def weights(self, current) -> np.ndarray:
        """Fisher weights 1 / sigma_log10^2."""
        s = self.log10_sigma(current)
        return 1.0 / np.maximum(s, 1e-30) ** 2

    def usable(self, current, snr: float = 3.0) -> np.ndarray:
        """Points whose current clears the noise floor by ``snr``."""
        return np.abs(np.asarray(current, dtype=float)) >= snr * self.i_floor

    def without_floor(self) -> "NoiseModel":
        return NoiseModel(self.sigma_rel, self.i_floor, floor_aware=False)
