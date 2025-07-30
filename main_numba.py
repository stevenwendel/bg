import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "0"

import numpy as np
from numba import njit, prange, float32, uint8
from src.constants import *
from time import time, perf_counter

N = len(NEURON_NAMES)
ALPHA_L = 250
td = np.arange(1, ALPHA_L + 1, dtype=np.float32)
alpha = (td / 30) * np.exp((30 - td) / 30)   

cue_wave = np.zeros(TMAX, dtype=np.float32)
go_wave = np.zeros_like(cue_wave)
cue_wave[EPOCHS['sample'][0]:EPOCHS['sample'][1]] = CUE_STRENGTH
go_wave[EPOCHS['response'][0]:EPOCHS['response'][0] + GO_DURATION] = GO_STRENGTH


# --------------------------------------------------------------------
# Hand‑crafted weights -------------------------------------------------
# --------------------------------------------------------------------
new_jh_weights = [
    ("Somat", "ALMprep", 40),
    ("Somat", "MSN1", 220),
    ("MSN1", "SNR1", -90),
    ("SNR1", "VMprep", -10),
    ("VMprep", "ALMprep", 70),
    ("ALMprep", "VMprep", 80),
    ("ALMprep", "MSN2", 320),
    ("MSN2", "SNR2", -50),
    ("SNR2", "VMresp", -100),
    ("PPN", "THALgo", 60),
    ("THALgo", "ALMinter", 55),
    ("ALMinter", "ALMprep", -50),
    ("THALgo", "ALMresp", 30),
    ("ALMresp", "MSN3", 320),
    ("MSN3", "SNR3", -90),
    ("SNR3", "VMresp", -50),
    ("VMresp", "ALMresp", 85),
    ("ALMresp", "VMresp", 90),
]

# --------------------------------------------------------------------
# Build weight matrix -------------------------------------------------
# --------------------------------------------------------------------
N = len(NEURON_NAMES)
W = np.zeros((N, N), dtype=np.float32)
for pre, post, w in new_jh_weights:
    i = NEURON_NAMES.index(pre)
    j = NEURON_NAMES.index(post)
    W[i, j] += w

pass_ids = [NEURON_NAMES.index(x) for x in ["VMresp", "ALMresp", "SNR3"]]
pass_ids = np.array(pass_ids)
print(pass_ids)

# CREATING CRITERION
conditions = []
for condition in CRITERIA:
    condition_criteria = []
    for neuron_name, neuron in CRITERIA[condition].items():
        idx = NEURON_NAMES.index(neuron_name)
        baseline = np.ones(TMAX, np.uint8) if neuron_name in TONICALLY_ACTIVE_NEURONS else np.zeros(TMAX, np.uint8)
        start = neuron["interval"][0]
        end = neuron["interval"][1]
        target_status = neuron["io"]
        # print(idx, neuron_name, baseline)
        for i in baseline:
            if target_status == "off":
                baseline[start:end] = 0
            elif target_status == "on":
                baseline[start:end] = 1

        baseline = baseline.reshape(TMAX//BIN_SIZE, BIN_SIZE)
        baseline = np.sum(baseline, axis=1,dtype=np.uint32)
        baseline = (baseline != 0).astype(np.uint8)

        condition_criteria.append((neuron_name, idx, baseline))
    condition_criteria = sorted(condition_criteria, key=lambda tup: tup[1])
    conditions.append(condition_criteria)

crit_Exp, crit_Cont = conditions

crit_indices = np.array([neu[1] for neu in crit_Cont])
crit_Exp = np.vstack([neu[2] for neu in crit_Exp])
crit_Cont = np.vstack([neu[2] for neu in crit_Cont])

# Takes the state at t and updates world to t+1. Returns spikes from step

@njit(parallel=False, fastmath=True, cache=True)
def step_kernel(V, U, Ibuf, t_ptr,
                a, b, vreset, d, k, vr, vt, vpeak, C, E, 
                W, alpha):
    n, L = V.size, alpha.size
    spk  = np.zeros(n, dtype=np.uint8)

    # integrate -------------------------------------------------------
    for i in range(n):
        I = Ibuf[t_ptr, i]
        dV  = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I + E[i]) / C[i]
        dU  = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
        V[i] += dV
        U[i] += dU
        if V[i] >= vpeak[i]:
            V[i]  = vreset[i]
            U[i] += d[i]
            spk[i] = 1          # Double check the formula to make sure it aint wonky

    # distribute PSC --------------------------------------------------
    if np.sum(spk) > 0:
        post_I = spk.astype(np.float32) @ W                   # dense GEMV
        t_next = (t_ptr + 1) % L
        for k_shift in range(L):
            Ibuf[(t_next + k_shift) % L, :] += post_I * alpha[k_shift]

    Ibuf[t_ptr,:] = 0.0
    return spk, (t_ptr + 1) % L


@njit(fastmath=True, cache=True)
def score_bin(curr_bin_results, crit_matrix, crit_indices, bin_idx, pass_ids):
    score = 0
    for i in range(len(crit_indices)):
        idx = crit_indices[i]
        if curr_bin_results[idx] == crit_matrix[i, bin_idx]:
            score += 1
        elif (bin_idx * BIN_SIZE > 3500) and (idx in pass_ids):
            score += 1
    return score


# ────────────────────────────────────────────────────────────────────
# 2.  Simulation + scoring
# ────────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def simulate(W, 
            a, b, vreset, d, k, vr, vt, vpeak, C, E, 
            alpha, cue_wave, go_wave, 
            crit_Exp, crit_Cont, crit_indices, pass_ids,
            tmax, 
            control, 
            return_full 
            ):
    
    W = np.ascontiguousarray(W)
    V = np.full(N, -60.0, np.float32)
    U = np.zeros_like(V, np.float32)
    Ibuf = np.zeros((ALPHA_L, N), dtype=np.float32)
    HIST = np.zeros((N, BIN_SIZE), np.uint8) # 99?
    if return_full:
        temp_full_hist = np.zeros((N, tmax), np.uint8) # 99?

    score = 0
    t_ptr   = 0
    bin = 0

    for t in range(tmax):

        if control == False:
            Ibuf[t_ptr,0] += cue_wave[t]
        Ibuf[t_ptr,7] += go_wave[t]
    
        spk, t_ptr = step_kernel(V, U, Ibuf, t_ptr,
                                 a, b, vreset, d, k, vr, vt, vpeak, C, E, 
                                 W, alpha)

        if return_full:
            temp_full_hist[:,t] = spk 

        cidx = t % BIN_SIZE
        HIST[:,cidx] = spk
        # bit-pack history
        if cidx == (BIN_SIZE - 1):
            curr_bin_results = (np.sum(HIST, axis=1) >= 1).astype(np.uint8)
            crits = crit_Exp if (control == False) else crit_Cont
            score += score_bin(curr_bin_results,crits, crit_indices, bin, pass_ids)
            bin += 1

    return score, (temp_full_hist if return_full else None)


start = perf_counter()
simulate(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
         alpha, cue_wave, go_wave,
         crit_Exp, crit_Cont, crit_indices, pass_ids,
         5000, False, False)

# print(mid-start)
# run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
#           alpha, cue_wave, go_wave,
#           crit_Exp, crit_Cont, crit_indices, pass_ids,
#           TMAX)
end = perf_counter()
print(f'Total time for single: {end - start:.3f}s')

@njit(cache=True)
def run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
              alpha, cue_wave, go_wave,
              crit_Exp, crit_Cont, crit_indices, pass_ids,
              tmax, NTRIALS=100):

    total_time = 0.0
    for i in range(NTRIALS):
        s1, _ = simulate(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
                         alpha, cue_wave, go_wave,
                         crit_Exp, crit_Cont, crit_indices, pass_ids,
                         tmax, False, False)
        s2, _ = simulate(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
                         alpha, cue_wave, go_wave,
                         crit_Exp, crit_Cont, crit_indices, pass_ids,
                         tmax, True, False)
    return s1, s2


start = perf_counter()
s1,s2=run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
          alpha, cue_wave, go_wave,
          crit_Exp, crit_Cont, crit_indices, pass_ids,
          TMAX, NTRIALS=1)
end = perf_counter()
print(f'Total time for 1: {end - start:.3f}s')

start = perf_counter()
s1,s2=run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
          alpha, cue_wave, go_wave,
          crit_Exp, crit_Cont, crit_indices, pass_ids,
          TMAX, NTRIALS=100)
end = perf_counter()
print(f'Total time for 100: {end - start:.3f}s')

start = perf_counter()
s1,s2=run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
          alpha, cue_wave, go_wave,
          crit_Exp, crit_Cont, crit_indices, pass_ids,
          TMAX, NTRIALS=200)
end = perf_counter()
print(f'Total time for 200: {end - start:.3f}s')

start = perf_counter()
s1,s2=run_batch(W, a, b, vreset, d, k, vr, vt, vpeak, C, E,
          alpha, cue_wave, go_wave,
          crit_Exp, crit_Cont, crit_indices, pass_ids,
          TMAX, NTRIALS=400)
end = perf_counter()
print(f'Total time for 200: {end - start:.3f}s')
