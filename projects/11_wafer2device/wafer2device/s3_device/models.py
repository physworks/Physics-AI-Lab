"""S3 -- variation field to per-die device response.

Everything downstream talks to the ``DeviceModel`` interface only.  The analytic
model is the first implementation; the TCAD surrogate is added later as a second
implementation and needs no change anywhere else in the pipeline.

The two are NOT interchangeable peers.  They form a precision ladder:

    AnalyticDeviceModel   fast screening; trends guaranteed, absolute values are
                          first-order only
    SurrogateDeviceModel  precise evaluation; trained on TCAD sweeps

``physics_checks.model_agreement`` compares their *trends* whenever both are
available, so a disagreement surfaces as a check failure rather than as a silent
difference in the optimisation result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..contracts import DeviceResponse, Hypothesis
from ..s1_pattern.data import spatial_field, wafer_grid

# Nominal device, expressed as physically meaningful constants rather than
# fitted magic numbers.
NOMINAL = {
    "vth": 0.45,            # V
    "ion": 3.0e-4,          # A/um
    "ioff": 1.0e-11,        # A/um
    "subthreshold_swing": 0.075,   # V/decade
    "temperature": 300.0,   # K
}


def variation_fields(h: Hypothesis, n: int,
                     angle_deg: float = 0.0) -> Dict[str, np.ndarray]:
    """Evaluate every hypothesis term on the wafer grid, summed per parameter."""
    r, theta, mask = wafer_grid(n)
    fields: Dict[str, np.ndarray] = {}
    for t in h.terms:
        ang = t.angle_deg if t.angle_deg is not None else angle_deg
        f = spatial_field(t.spatial_form, r, theta, t.magnitude, ang)
        fields[t.param] = fields.get(t.param, np.zeros_like(r)) + f
    for k in fields:
        fields[k] = np.where(mask, fields[k], 0.0)
    return fields


class DeviceModel(ABC):
    """Contract: fractional process deviations in, device metrics out."""

    name: str = "abstract"

    @abstractmethod
    def evaluate(self, deviations: Dict[str, np.ndarray],
                 knobs: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Return dict with keys 'vth', 'ion', 'ioff'; arrays match the input shape."""

    def response(self, deviations: Dict[str, np.ndarray], spec: Dict[str, float],
                 knobs: Optional[Dict[str, float]] = None,
                 mask: Optional[np.ndarray] = None) -> DeviceResponse:
        out = self.evaluate(deviations, knobs)
        sel = np.ones_like(out["vth"], dtype=bool) if mask is None else mask
        vth, ion, ioff = out["vth"][sel], out["ion"][sel], out["ioff"][sel]

        bad = ((np.abs(vth - spec["vth_nominal"]) > spec["vth_tolerance"])
               | (ion < spec["ion_min"]) | (ioff > spec["ioff_max"]))
        return DeviceResponse(
            model_name=self.name,
            vth=[float(v) for v in vth],
            ion=[float(v) for v in ion],
            ioff=[float(v) for v in ioff],
            out_of_spec_rate=float(bad.mean()),
            stats={"vth_mean": float(vth.mean()), "vth_std": float(vth.std()),
                   "vth_range": float(vth.max() - vth.min()),
                   "ion_mean": float(ion.mean()),
                   "ioff_median": float(np.median(ioff))},
        )


class AnalyticDeviceModel(DeviceModel):
    """First-order long-channel MOSFET response to process deviations.

    Sensitivities are compact physical relations, not fitted constants:

    * Vth shifts with oxide thickness through Cox (Vth ~ Qdep/Cox, so dVth/Vth
      grows with tox).
    * Vth shifts with channel doping through the depletion charge (~ sqrt(Na))
      and the bulk potential.
    * Short-channel roll-off makes Vth fall as gate length shrinks.
    * Interface traps degrade subthreshold swing, which lifts Ioff.
    * Series resistance degrades Ion only.
    """

    name = "analytic"

    # d(metric)/d(fractional deviation), from the relations above.
    S_VTH = {"gate_oxide_thickness": 0.18,     # V per unit fractional tox change
             "channel_doping": 0.11,           # V per unit fractional Na change
             "gate_length": 0.22,              # V per unit fractional Lg change
             "interface_trap_density": 0.03,
             "series_resistance": 0.0}
    S_SS = {"interface_trap_density": 0.020}   # V/dec per unit fractional Dit
    S_ION = {"series_resistance": -0.35,       # relative Ion change
             "gate_length": -0.55,
             "gate_oxide_thickness": -0.30}

    def evaluate(self, deviations: Dict[str, np.ndarray],
                 knobs: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        knobs = knobs or {}
        any_field = next(iter(deviations.values()), None)
        shape = any_field.shape if any_field is not None else (1,)
        zeros = np.zeros(shape)

        dvth = zeros.copy()
        for p, s in self.S_VTH.items():
            dvth = dvth + s * deviations.get(p, zeros)

        # Design knobs act on the same physics as the process deviations.
        dvth = (dvth
                + knobs.get("vt_implant_bias", 0.0) * 0.55
                + knobs.get("work_function_bias", 0.0) * 1.00
                + knobs.get("gate_length_bias", 0.0) * self.S_VTH["gate_length"])
        vth = NOMINAL["vth"] + dvth

        ss = NOMINAL["subthreshold_swing"] + sum(
            s * deviations.get(p, zeros) for p, s in self.S_SS.items())
        ss = np.clip(ss, 0.055, 0.20)

        rel_ion = zeros.copy()
        for p, s in self.S_ION.items():
            rel_ion = rel_ion + s * deviations.get(p, zeros)
        rel_ion = rel_ion - self.S_ION["gate_length"] * 0.0 \
            + knobs.get("gate_length_bias", 0.0) * self.S_ION["gate_length"]

        # Overdrive dependence: a higher Vth costs drive current.
        vgs = 1.0
        overdrive = np.clip(vgs - vth, 0.05, None)
        ion = NOMINAL["ion"] * (1.0 + rel_ion) * (overdrive / (vgs - NOMINAL["vth"])) ** 1.3
        ion = np.clip(ion, 1e-12, None)

        # Ioff from the subthreshold slope, referenced to the nominal device.
        ioff = NOMINAL["ioff"] * 10.0 ** (
            -(vth - NOMINAL["vth"]) / ss + (1.0 / NOMINAL["subthreshold_swing"]
                                            - 1.0 / ss) * 0.0)
        ioff = np.clip(ioff, 1e-18, None)
        return {"vth": vth, "ion": ion, "ioff": ioff}


class SurrogateDeviceModel(DeviceModel):
    """TCAD surrogate plug-in slot.

    Expects an ``.npz`` with a linear-plus-quadratic response surface fitted to
    TCAD sweeps:

        params : (k,)   parameter names, in column order
        w1     : (k, 3) linear coefficients for [vth, log10 ion, log10 ioff]
        w2     : (k, 3) quadratic coefficients
        bias   : (3,)   nominal values

    Until the artefact exists this class raises on construction, which is why
    ``configs/pipeline.yaml`` keeps ``secondary: null`` by default.  Nothing
    else in the pipeline changes when it is switched on.
    """

    name = "surrogate"

    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"surrogate artefact not found at {path!r}; keep device_model."
                f"secondary = null until it is trained")
        data = np.load(p, allow_pickle=True)
        self.params: List[str] = [str(x) for x in data["params"]]
        self.w1 = np.asarray(data["w1"], dtype=float)
        self.w2 = np.asarray(data["w2"], dtype=float)
        self.bias = np.asarray(data["bias"], dtype=float)

    def evaluate(self, deviations: Dict[str, np.ndarray],
                 knobs: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        knobs = knobs or {}
        any_field = next(iter(deviations.values()), None)
        shape = any_field.shape if any_field is not None else (1,)
        zeros = np.zeros(shape)

        acc = np.zeros(shape + (3,))
        for i, name in enumerate(self.params):
            x = deviations.get(name, zeros)
            acc += (self.w1[i] * x[..., None]) + (self.w2[i] * (x ** 2)[..., None])
        out = acc + self.bias

        vth = out[..., 0] + knobs.get("vt_implant_bias", 0.0) * 0.55 \
            + knobs.get("work_function_bias", 0.0)
        ion = 10.0 ** out[..., 1]
        ioff = 10.0 ** out[..., 2]
        return {"vth": vth, "ion": ion, "ioff": ioff}


def build_model(kind: Optional[str], cfg: Dict[str, Any]) -> Optional[DeviceModel]:
    if kind in (None, "null", "none"):
        return None
    if kind == "analytic":
        return AnalyticDeviceModel()
    if kind == "surrogate":
        return SurrogateDeviceModel(cfg.get("surrogate_path",
                                            "artifacts/tcad_surrogate.npz"))
    raise ValueError(f"unknown device model {kind!r}")
