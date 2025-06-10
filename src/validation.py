"""Fast criteria construction and scoring utilities.

Exports
-------
* ``define_criteria()`` – returns the pre‑built ON/OFF masks (dict).
* ``evaluate_conditions(raster_ms)`` – given a 0/1 spike raster (N×TMAX)
  returns experimental/control scores.

All operations are pure NumPy; cost < 1 ms.
"""
from __future__ import annotations

import numpy as np
from src.constants import (
    CRITERIA, CRITERIA_NAMES, TONICALLY_ACTIVE_NEURONS,
    BIN_SIZE, TMAX, NEURON_NAMES,
)

__all__ = ["define_criteria", "evaluate_conditions", "evaluate_dna"]

# --------------------------------------------------------------------
# 1.  Pre‑compute helper indices & constants
# --------------------------------------------------------------------
_NUM_PERIODS = (TMAX + BIN_SIZE - 1) // BIN_SIZE
_CRIT_ROW    = np.array([NEURON_NAMES.index(n) for n in CRITERIA_NAMES], dtype=np.intp)

# --------------------------------------------------------------------
# 2.  Build criteria masks once at import time
# --------------------------------------------------------------------

def _build_mask(defn: dict[str, dict]) -> np.ndarray:
    mask = np.zeros((_NUM_PERIODS, len(CRITERIA_NAMES)), np.uint8)
    for row, n in enumerate(CRITERIA_NAMES):
        if n in TONICALLY_ACTIVE_NEURONS:
            mask[:, row] = 1
    for row, n in enumerate(CRITERIA_NAMES):
        meta = defn[n]
        p0 = meta["interval"][0] // BIN_SIZE
        p1 = min(meta["interval"][1] // BIN_SIZE, _NUM_PERIODS)
        mask[p0:p1, row] = 1 if meta["io"] == "on" else 0
    return mask.T  # rows × periods

_CRIT_MASKS = {k: _build_mask(v) for k, v in CRITERIA.items()}

# --------------------------------------------------------------------
# 3.  Raster binning
# --------------------------------------------------------------------

def _bin_raster(ms_raster: np.ndarray) -> np.ndarray:
    trimmed = ms_raster[:, : _NUM_PERIODS * BIN_SIZE]
    return trimmed.reshape(trimmed.shape[0], _NUM_PERIODS, BIN_SIZE).sum(axis=2)

# --------------------------------------------------------------------
# 4.  Scoring
# --------------------------------------------------------------------

def _score(binned: np.ndarray, mask: np.ndarray, halve: bool) -> int:
    match = (binned > 0) == mask
    try:
        alm = CRITERIA_NAMES.index("ALMresp")
        p4000 = 4000 // BIN_SIZE
        match[alm, p4000:] = True
    except ValueError:
        pass
    val = int(match.sum())
    return val // 2 if halve else val

# --------------------------------------------------------------------
# 5.  Public API
# --------------------------------------------------------------------

def define_criteria() -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in _CRIT_MASKS.items()}


def evaluate_conditions(ms_raster: np.ndarray) -> dict[str, int]:
    binned = _bin_raster(ms_raster)[_CRIT_ROW]
    return {
        "experimental": _score(binned, _CRIT_MASKS["experimental"], False),
        "control": _score(binned, _CRIT_MASKS["control"], True),
    }

# --------------------------------------------------------------------
# 6.  Legacy wrapper (GA code expects evaluate_dna)
# --------------------------------------------------------------------

def evaluate_dna(*, dna_matrix, neurons, alpha_array, input_waves, criteria, **kw):
    from src.neuron import _SPIKES, t_pointer as _tp, prepare_neurons
    from src.network import run_network

    cue_wave, go_wave = input_waves
    out = {}
    for label, ctl in (("experimental", False), ("control", True)):
        prepare_neurons(neurons, cue_wave, go_wave, ctl)
        _tp = 0
        run_network(neurons, dna_matrix, alpha_array)
        out[label] = evaluate_conditions(_SPIKES)[label]
    return out, {}
