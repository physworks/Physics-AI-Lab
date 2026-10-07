"""Circuit elements and the residual they contribute.

Modified nodal analysis, solved by Newton on the residual rather than by
stamping companion models.  The unknown vector is

    x = [ node voltages (ground excluded) , voltage-source currents ]

and each element adds to the KCL residual ``F`` and to the Jacobian ``J``.
Writing it as a residual keeps the device's own derivatives -- ``gm`` and
``gds`` straight out of ``device.evaluate`` -- in the Jacobian unmodified,
which matters here because the whole point of the stage is to find out what
those derivatives do to a solver.

Ground is node 0 and carries no equation; a contribution to it is dropped.

The compact model is **NMOS only**, so every circuit here is NMOS-with-load.
There is no PMOS and no CMOS inverter, and pretending otherwise would be the
kind of claim this repository does not make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .device import DeviceParams, evaluate

GROUND = 0


class Element:
    """Base: contribute to the KCL residual and its Jacobian."""

    def n_extra(self) -> int:
        return 0

    def stamp(self, x: np.ndarray, F: np.ndarray, J: np.ndarray,
              nmap: Dict[str, int], extra_row: int) -> None:
        raise NotImplementedError


def _add(F, J, nmap, node, value, jac_terms):
    """Add ``value`` to node's residual and its Jacobian row, skipping ground."""
    i = nmap[node]
    if i < 0:
        return
    F[i] += value
    for other, d in jac_terms:
        k = nmap[other] if isinstance(other, str) else other
        if k >= 0:
            J[i, k] += d


@dataclass
class Resistor(Element):
    a: str
    b: str
    r: float
    name: str = "R"

    def stamp(self, x, F, J, nmap, extra_row):
        g = 1.0 / self.r
        va = x[nmap[self.a]] if nmap[self.a] >= 0 else 0.0
        vb = x[nmap[self.b]] if nmap[self.b] >= 0 else 0.0
        i = (va - vb) * g
        _add(F, J, nmap, self.a, +i, [(self.a, +g), (self.b, -g)])
        _add(F, J, nmap, self.b, -i, [(self.a, -g), (self.b, +g)])


@dataclass
class CurrentSource(Element):
    """Constant current forced from node ``a`` to node ``b``."""

    a: str
    b: str
    i: float
    name: str = "I"

    def stamp(self, x, F, J, nmap, extra_row):
        _add(F, J, nmap, self.a, +self.i, [])
        _add(F, J, nmap, self.b, -self.i, [])


@dataclass
class VoltageSource(Element):
    p: str
    n: str
    v: float
    name: str = "V"

    def n_extra(self) -> int:
        return 1

    def stamp(self, x, F, J, nmap, extra_row):
        ik = x[extra_row]
        _add(F, J, nmap, self.p, +ik, [(extra_row, +1.0)])
        _add(F, J, nmap, self.n, -ik, [(extra_row, -1.0)])

        vp = x[nmap[self.p]] if nmap[self.p] >= 0 else 0.0
        vn = x[nmap[self.n]] if nmap[self.n] >= 0 else 0.0
        F[extra_row] += vp - vn - self.v
        for node, sign in ((self.p, +1.0), (self.n, -1.0)):
            k = nmap[node]
            if k >= 0:
                J[extra_row, k] += sign


@dataclass
class Capacitor(Element):
    """Backward-Euler companion; inert unless ``dt`` is set by the solver."""

    a: str
    b: str
    c: float
    name: str = "C"
    dt: float | None = field(default=None, repr=False)
    v_old: float = field(default=0.0, repr=False)

    def stamp(self, x, F, J, nmap, extra_row):
        if self.dt is None:
            return
        g = self.c / self.dt
        va = x[nmap[self.a]] if nmap[self.a] >= 0 else 0.0
        vb = x[nmap[self.b]] if nmap[self.b] >= 0 else 0.0
        i = g * ((va - vb) - self.v_old)
        _add(F, J, nmap, self.a, +i, [(self.a, +g), (self.b, -g)])
        _add(F, J, nmap, self.b, -i, [(self.a, -g), (self.b, +g)])

    def voltage(self, x, nmap) -> float:
        va = x[nmap[self.a]] if nmap[self.a] >= 0 else 0.0
        vb = x[nmap[self.b]] if nmap[self.b] >= 0 else 0.0
        return float(va - vb)


@dataclass
class Nmos(Element):
    """The compact model as a three-terminal element.

    ``gm`` and ``gds`` come from ``device.evaluate``, which solves the series
    resistance to tolerance and differentiates it analytically.  Nothing is
    smoothed or clipped on the way into the Jacobian: if the model hands the
    solver a negative output conductance, the solver gets a negative output
    conductance, which is what this stage is here to observe.
    """

    d: str
    g: str
    s: str
    params: DeviceParams
    name: str = "M"
    last: Dict[str, float] = field(default_factory=dict, repr=False)

    def stamp(self, x, F, J, nmap, extra_row):
        vd = x[nmap[self.d]] if nmap[self.d] >= 0 else 0.0
        vg = x[nmap[self.g]] if nmap[self.g] >= 0 else 0.0
        vs = x[nmap[self.s]] if nmap[self.s] >= 0 else 0.0

        ev = evaluate(self.params, np.array([vg - vs]), np.array([vd - vs]))
        ids = float(ev.ids[0])
        gm = float(ev.gm[0])
        gds = float(ev.gds[0])
        self.last = {"ids": ids, "gm": gm, "gds": gds,
                     "vgs": vg - vs, "vds": vd - vs,
                     "inner_iterations": ev.iterations,
                     "inner_converged": ev.converged}

        _add(F, J, nmap, self.d, +ids,
             [(self.d, +gds), (self.g, +gm), (self.s, -(gm + gds))])
        _add(F, J, nmap, self.s, -ids,
             [(self.d, -gds), (self.g, -gm), (self.s, +(gm + gds))])


@dataclass
class Circuit:
    elements: List[Element] = field(default_factory=list)
    _nodes: List[str] = field(default_factory=lambda: ["0"])

    def add(self, el: Element) -> Element:
        for attr in ("a", "b", "p", "n", "d", "g", "s"):
            node = getattr(el, attr, None)
            if isinstance(node, str) and node not in self._nodes:
                self._nodes.append(node)
        self.elements.append(el)
        return el

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    def node_map(self) -> Dict[str, int]:
        """Node name to index; ground maps to -1 and is dropped."""
        return {n: (i - 1) for i, n in enumerate(self._nodes)}

    def size(self) -> tuple[int, int]:
        n = len(self._nodes) - 1
        m = sum(e.n_extra() for e in self.elements)
        return n, m

    def residual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, m = self.size()
        F = np.zeros(n + m)
        J = np.zeros((n + m, n + m))
        nmap = self.node_map()
        row = n
        for el in self.elements:
            el.stamp(x, F, J, nmap, row)
            row += el.n_extra()
        return F, J

    def devices(self) -> List[Nmos]:
        return [e for e in self.elements if isinstance(e, Nmos)]

    def capacitors(self) -> List[Capacitor]:
        return [e for e in self.elements if isinstance(e, Capacitor)]

    def voltage(self, x: np.ndarray, node: str) -> float:
        i = self.node_map()[node]
        return 0.0 if i < 0 else float(x[i])
