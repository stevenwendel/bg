"""Network utilities – experiment builder and fast simulation loop.

The `run_network` function now fetches `_INPUT`, `_VHIST`, `_vpeak` from
`src.neuron` **at call‑time**, not at import‑time.  This prevents the
`NoneType` crash in multiprocessing workers where the neuron arrays are
allocated *after* the module is imported.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.constants import TMAX, BIN_SIZE, NEURON_NAMES
from src.neuron import vectorised_step

# # --------------------------------------------------------------------
# # Alpha‑function PSP adder (Numba)
# # --------------------------------------------------------------------

# @njit(parallel=True, fastmath=True, cache=True)
# def _add_psp(input_buf: np.ndarray, post_I: np.ndarray, alpha: np.ndarray,
#              start: int, lend: int):
#     """In‑place add of alpha‑shaped current starting at `start+1`."""
#     n = post_I.size
#     for j in prange(n):
#         if post_I[j] == 0.0:
#             continue
#         base = start + 1
#         for k in range(lend):
#             input_buf[j, base + k] += post_I[j] * alpha[k]

# --------------------------------------------------------------------
# Public helpers (create_experiment unchanged)
# --------------------------------------------------------------------

def create_experiment():
    n_bins = TMAX / BIN_SIZE
    assert n_bins.is_integer(), "TMAX must be divisible by BIN_SIZE"
    periods = np.linspace(0, TMAX, int(n_bins) + 1)

    from src.constants import EPOCHS, CUE_STRENGTH, GO_STRENGTH, GO_DURATION
    sq = np.zeros(TMAX, np.float32)
    go = np.zeros_like(sq)
    sq[EPOCHS['sample'][0]:EPOCHS['sample'][1]] = CUE_STRENGTH
    go[EPOCHS['response'][0]:EPOCHS['response'][0] + GO_DURATION] = GO_STRENGTH
    input_waves = [sq, go]

    from src.utils import create_alpha_array
    alpha = create_alpha_array(250, L=30)
    return periods, input_waves, alpha

# --------------------------------------------------------------------
# Fast runner – arrays fetched at call‑time
# --------------------------------------------------------------------

def run_network(neurons, weight_matrix: np.ndarray, alpha_kernel: np.ndarray):
    import src.neuron as n  # ensure we reference *current* arrays

    N = len(neurons)
    L = alpha_kernel.size
    spikers = np.zeros(N, np.uint8)

    for t in range(TMAX):
        # inside run_network (before the Euler step)
        if spikers.any():
            post_I = spikers.astype(np.float32) @ weight_matrix        # (N,)
            lend   = min(L, TMAX - t - 1)
            n._INPUT[:, t+1:t+1+lend] += post_I[:, None] * alpha_kernel[:lend]


        spikers = vectorised_step(n._INPUT[:, t])

        if t:
            peaked = spikers.astype(bool)
            n._VHIST[peaked, t - 1] = n._vpeak[peaked]

    return spikers
