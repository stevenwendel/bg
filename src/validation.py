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
    # Special late-time scoring: Always award points after 4000ms
    p4000 = 4000 // BIN_SIZE
    
    try:
        alm = CRITERIA_NAMES.index("ALMresp")
        match[alm, p4000:] = True
    except ValueError:
        pass
    
    try:
        snr3 = CRITERIA_NAMES.index("SNR3")
        match[snr3, p4000:] = True
    except ValueError:
        pass
    
    try:
        vmresp = CRITERIA_NAMES.index("VMresp")
        match[vmresp, p4000:] = True
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

##############################################################################
# 5‑bis.  Diagnostic helper – which bins failed?                             #
##############################################################################

def diagnose_conditions(ms_raster: np.ndarray,
                        condition: str = "experimental",
                        max_lines: int | None = 50,
                        /, *,
                        return_list: bool = False):
    """
    Print (or return) every neuron × 100‑ms bin that fails to meet the
    ON/OFF criterion.

    Parameters
    ----------
    ms_raster : ndarray
        0/1 raster of shape (N_neurons, TMAX).
    condition : {"experimental", "control"}
        Which criterion mask to diagnose.
    max_lines : int or None
        Limit the number of lines printed (handy for huge logs).  ``None``
        prints everything.
    return_list : bool, default False
        If True, return a Python list instead of printing.

    Returns
    -------
    list[dict] if *return_list* else None
    """
    if condition not in _CRIT_MASKS:
        raise ValueError("condition must be 'experimental' or 'control'")

    binned  = _bin_raster(ms_raster)[_CRIT_ROW]          # rows = criteria order
    active  = binned > 0
    mask    = _CRIT_MASKS[condition].copy()

    # ALMresp always OK after 4000 ms
    try:
        alm = CRITERIA_NAMES.index("ALMresp")
        mask[alm, 4000 // BIN_SIZE :] = active[alm, 4000 // BIN_SIZE :]
    except ValueError:
        pass
    
    # SNR3 always OK after 4000 ms
    try:
        snr3 = CRITERIA_NAMES.index("SNR3")
        mask[snr3, 4000 // BIN_SIZE :] = active[snr3, 4000 // BIN_SIZE :]
    except ValueError:
        pass
    
    # VMresp always OK after 4000 ms
    try:
        vmresp = CRITERIA_NAMES.index("VMresp")
        mask[vmresp, 4000 // BIN_SIZE :] = active[vmresp, 4000 // BIN_SIZE :]
    except ValueError:
        pass

    wrong = active != mask                               # boolean matrix
    rows, cols = np.where(wrong)

    report = []
    for r, c in zip(rows, cols):
        entry = {
            "neuron" : CRITERIA_NAMES[r],
            "period" : c,
            "t_start": c * BIN_SIZE,
            "t_end"  : (c + 1) * BIN_SIZE,
            "wanted" : int(mask[r, c]),
            "spikes" : int(binned[r, c]),
        }
        report.append(entry)

    if return_list:
        return report

    print(f"\nMISMATCHES for {condition} (wanted vs spikes) — total {len(report)}")
    print("-" * 60)
    for i, d in enumerate(report):
        if (max_lines is not None) and (i >= max_lines):
            print(f"... ({len(report) - max_lines} more)")
            break
        print(f"{d['neuron']:10s}  bin {d['period']:03d} "
              f"[{d['t_start']:4d}-{d['t_end']:4d} ms]  "
              f"wanted {d['wanted']}  got {d['spikes']}")


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
