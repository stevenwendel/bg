from numba import njit, prange, float32, uint8
import numpy as np


@njit(parallel=True, fastmath=True, cache=True)
def step_kernel(V, U, Ibuf, W_pre2post, alpha, t_ptr, a,b,k,vr,vt,vpeak,vreset,d,C):
    par = PARAM_RS if TYPE_IDX[i]==0 else PARAM_MSN
    
    a,b,k,vr,vt,vpeak,vreset,d,C = par
    
    n, L = V.size, alpha.size
    spk  = np.zeros(n, dtype=uint8)

    # ----- 1.  integrate Izhikevich ---------------------------------
    for i in prange(n):
        I = Ibuf[i, t_ptr]              # current bin  ⟵ O(1)
        dV = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I) / C[i]
        dU = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
        V[i] += dV
        U[i] += dU
        if V[i] >= vpeak[i]:
            V[i]  = vreset[i]
            U[i] += d[i]
            spk[i] = 1

    # ----- 2.  distribute PSP to *future* slots ---------------------
    if spk.any():
        post_I = spk.astype(float32) @ W_pre2post          # (N,) BLAS
        t_next = (t_ptr + 1) % L
        for k_shift in range(alpha.size):                  # tiny loop, L≤250
            Ibuf[:, (t_next + k_shift) % L] += post_I * alpha[k_shift]

    # zero out the slot we just used (makes buffer circular)
    Ibuf[:, t_ptr] = 0.0

    # advance circular pointer
    t_ptr = (t_ptr + 1) % L
    return spk, t_ptr


def simulate(W, cue_wave, go_wave, alpha, T=5000):
    V,U = V0.copy(), U0.copy()          # const templates
    Ibuf= np.zeros((N, alpha.size), np.float32)
    t_ptr = 0
    exp_score = ctrl_score = 0

    for ctl in (False, True):
        # inject exogenous currents
        if ctl:
            Ibuf[ppn_idx, cue_on:cue_off] += go_wave  # example
        else:
            Ibuf[somat_idx, sample_on:sample_off] += cue_wave

        for _ in range(T):
            spk, t_ptr = step_kernel(
                V,U,Ibuf,W,alpha,t_ptr,a,b,k,vr,vt,vpeak,vreset,d,C
            )
            # epoch-based scoring here with simple counters → O(1)

        if ctl:
            ctrl_score = current_score
        else:
            exp_score  = current_score
    return exp_score + ctrl_score//2

if __name__ == "__main__":
    pass
    # for gen in range(NUM_GEN):
        # fitness = [simulate(decode(d), cue, go, alpha) for d in pop]