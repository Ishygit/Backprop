"""
Case 1: Oven calibration, random-walk Metropolis on (theta, phi).

Model
    phi         ~ Uniform(160, 240)      setpoint (deg C)
    theta | phi ~ Normal(phi, 5^2)       actual temperature
    y | theta   ~ Normal(theta, 3^2)     probe reading

Exact posterior (flat prior far from the data, so truncation is negligible):
    theta | y   ~ Normal(y, 3^2)
    phi | theta ~ Normal(theta, 5^2)
    phi | y     ~ Normal(y, 3^2 + 5^2)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate

# ---------------- Constants to play with ----------------
N_CHAINS       = 4        # number of repeat runs (independent chains)
BURN_IN        = 2_000    # steps discarded at the start of each chain
N_STEPS        = 50_000   # retained MCMC steps per chain (after burn-in)
BOX_HALF_WIDTH = 1.0      # neighborhood half-width (deg C): box is 2*h by 2*h
PROPOSAL_SD    = 4.0      # random-walk step size (try 0.1, 4, 50 to see effects)
SEED           = 42

# ---------------- Problem setup ----------------
Y_OBS = 212.0
PHI_LO, PHI_HI = 160.0, 240.0
SIGMA_THETA = 5.0         # sd of theta given phi
SIGMA_Y     = 3.0         # sd of probe noise
POINT = (214.0, 210.0)    # (theta, phi) at which to estimate posterior density


def log_target(theta, phi):
    """Log of the unnormalized posterior: likelihood * conditional prior * prior."""
    if not (PHI_LO < phi < PHI_HI):
        return -np.inf    # zero density outside the uniform's support
    return (stats.norm.logpdf(Y_OBS, theta, SIGMA_Y)
            + stats.norm.logpdf(theta, phi, SIGMA_THETA))
    # The uniform prior on phi is constant inside the support, so it drops out.


def run_chain(rng, start):
    n_total = BURN_IN + N_STEPS
    samples = np.empty((n_total, 2))
    state = np.array(start, dtype=float)
    logp = log_target(*state)
    accepted = 0
    for i in range(n_total):
        proposal = state + rng.normal(0.0, PROPOSAL_SD, size=2)   # symmetric
        logp_new = log_target(*proposal)
        if np.log(rng.uniform()) < logp_new - logp:
            state, logp = proposal, logp_new
            accepted += 1
        samples[i] = state
    return samples[BURN_IN:], accepted / n_total


def box_stats(samples):
    t, p = samples[:, 0], samples[:, 1]
    h = BOX_HALF_WIDTH
    inside = (abs(t - POINT[0]) < h) & (abs(p - POINT[1]) < h)
    frac = inside.mean()
    return frac, frac / (2 * h) ** 2      # box probability, density estimate


def exact_box_probability():
    h = BOX_HALF_WIDTH
    def integrand(t):
        pdf_t = stats.norm.pdf(t, Y_OBS, SIGMA_Y)
        p_phi = (stats.norm.cdf(POINT[1] + h, t, SIGMA_THETA)
                 - stats.norm.cdf(POINT[1] - h, t, SIGMA_THETA))
        return pdf_t * p_phi
    val, _ = integrate.quad(integrand, POINT[0] - h, POINT[0] + h)
    return val


def exact_density(t, p):
    return stats.norm.pdf(t, Y_OBS, SIGMA_Y) * stats.norm.pdf(p, t, SIGMA_THETA)


def autocorr(x, max_lag=200):
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    acf = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return acf / acf[0]


def main():
    rng = np.random.default_rng(SEED)
    # Overdispersed starting points, all inside the support of phi
    starts = [(Y_OBS + rng.uniform(-30, 30), Y_OBS + rng.uniform(-30, 30))
              for _ in range(N_CHAINS)]

    chains, rates = [], []
    for s in starts:
        c, r = run_chain(rng, s)
        chains.append(c)
        rates.append(r)

    # ---------- Report ----------
    exact_p = exact_box_probability()
    exact_d = exact_density(*POINT)
    fracs = np.array([box_stats(c)[0] for c in chains])
    pooled = np.vstack(chains)
    pooled_frac, pooled_dens = box_stats(pooled)

    print(f"Settings: chains={N_CHAINS}, burn-in={BURN_IN}, steps={N_STEPS}, "
          f"box half-width={BOX_HALF_WIDTH}, proposal sd={PROPOSAL_SD}")
    print(f"Acceptance rates: {np.round(rates, 3)}  (roughly 0.2-0.4 is a good target)\n")
    print(f"Box around (theta={POINT[0]}, phi={POINT[1]}):")
    print("  Per-chain box probability:", np.round(fracs, 5))
    print(f"  Spread across chains (sd): {fracs.std(ddof=1):.5f}")
    print(f"  Pooled MCMC estimate:      {pooled_frac:.5f}")
    print(f"  Exact box probability:     {exact_p:.5f}")
    print(f"  Pooled density estimate:   {pooled_dens:.5f}")
    print(f"  Exact density at point:    {exact_d:.5f}\n")
    print(f"theta: MCMC mean {pooled[:,0].mean():.2f}, sd {pooled[:,0].std():.2f}  "
          f"(exact {Y_OBS:.2f}, {SIGMA_Y:.2f})")
    print(f"phi:   MCMC mean {pooled[:,1].mean():.2f}, sd {pooled[:,1].std():.2f}  "
          f"(exact {Y_OBS:.2f}, {np.hypot(SIGMA_Y, SIGMA_THETA):.2f})")
    print(f"corr(theta, phi): MCMC {np.corrcoef(pooled.T)[0,1]:.3f}  "
          f"(exact {SIGMA_Y/np.hypot(SIGMA_Y, SIGMA_THETA):.3f})")

    # ---------- Plots ----------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # Joint scatter with exact contours and the box
    a = ax[0, 0]
    sub = pooled[::20]
    a.scatter(sub[:, 0], sub[:, 1], s=2, alpha=0.3)
    tg = np.linspace(200, 224, 200)
    pg = np.linspace(190, 234, 200)
    T, P = np.meshgrid(tg, pg)
    a.contour(T, P, exact_density(T, P), colors="k", linewidths=0.8)
    h = BOX_HALF_WIDTH
    a.add_patch(plt.Rectangle((POINT[0] - h, POINT[1] - h), 2 * h, 2 * h,
                              fill=False, color="red", lw=2))
    a.set(xlabel="theta (actual temp)", ylabel="phi (setpoint)",
          title="Joint posterior: samples vs exact contours")

    # Marginals
    a = ax[0, 1]
    a.hist(pooled[:, 0], bins=80, density=True, alpha=0.5)
    x = np.linspace(200, 224, 300)
    a.plot(x, stats.norm.pdf(x, Y_OBS, SIGMA_Y), "k")
    a.set(xlabel="theta", title="Marginal of theta")

    a = ax[0, 2]
    a.hist(pooled[:, 1], bins=80, density=True, alpha=0.5)
    x = np.linspace(190, 234, 300)
    a.plot(x, stats.norm.pdf(x, Y_OBS, np.hypot(SIGMA_Y, SIGMA_THETA)), "k")
    a.set(xlabel="phi", title="Marginal of phi")

    # Trace of the first 2000 retained steps, phi
    a = ax[1, 0]
    for c in chains:
        a.plot(c[:2000, 1], lw=0.6)
    a.set(xlabel="step", ylabel="phi", title="Trace (first 2000 retained steps)")

    # Autocorrelation
    a = ax[1, 1]
    for c in chains:
        a.plot(autocorr(c[:, 1]), lw=0.8)
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="lag", ylabel="autocorrelation", title="Autocorrelation of phi")

    # Box probability per chain
    a = ax[1, 2]
    a.bar(range(1, N_CHAINS + 1), fracs)
    a.axhline(exact_p, color="red", label="exact")
    a.set(xlabel="chain", ylabel="box probability",
          title="Box probability by chain")
    a.legend()

    plt.tight_layout()
    plt.savefig("oven_mcmc.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
