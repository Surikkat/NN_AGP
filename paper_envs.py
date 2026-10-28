"""Environment generators for Weak/Balanced, Smooth, Exp. 1–10 and KuaiRec."""
import itertools
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RHO, SD_ETA = 0.9, 0.1
SIG = SD_ETA / math.sqrt(1 - RHO**2)
DATA = Path(__file__).resolve().parent / "datasets"
KUAIREC = "https://zenodo.org/records/18164998/files/KuaiRec.zip"


class WeakBalancedEnv:
    """Gaussian contexts; context baseline plus two RBF action preference factors."""
    def __init__(self, spec, seed, horizon):
        self.spec = spec
        env_rng, context_rng, noise_rng = [np.random.default_rng(s)
            for s in np.random.SeedSequence([seed, 401]).spawn(3)]
        d, K = spec["d_theta"], spec["actions"]
        self.rotation = np.linalg.qr(env_rng.normal(size=(d, d)))[0]
        self.X = np.linspace(-2., 2., K)[:, None]
        kernel = np.exp(-.5 * (self.X-self.X.T)**2)
        functions = np.linalg.cholesky(kernel+1e-10*np.eye(K)) @ env_rng.normal(size=(K, 2))
        functions -= functions.mean(0)
        functions /= np.sqrt(np.mean(functions**2, axis=0))
        self.preferences = functions
        raw = self.raw_contexts(context_rng, horizon)
        self.theta = raw @ self.rotation
        self.rewards = (spec["baseline"]*raw[:, :1]
                        + raw[:, 1:2]*functions[:, 0]
                        + spec["weak"]*raw[:, 2:3]*functions[:, 1])
        self.noise = spec["noise"]*noise_rng.normal(size=horizon)

    def raw_contexts(self, rng, n):
        z = rng.normal(size=(n, self.spec["d_theta"]))
        rho = self.spec["nuisance_correlation"]
        if rho:
            z[:, 3:] = np.sqrt(1-rho)*z[:, 3:] + np.sqrt(rho)*rng.normal(size=(n, 1))
        return z

    def calibration(self, seed, t, n):
        rng = np.random.default_rng([seed, 90211, t])
        return self.raw_contexts(rng, n) @ self.rotation


def hermite(u, D):
    """Normalized Hermite products, ordered by degree 1..D then lexicographically."""
    he = [np.ones_like(u), u]
    for k in range(1, D):
        he.append(u * he[k] - k * he[k - 1])
    cols = []
    for deg in range(1, D + 1):
        for idx in itertools.combinations_with_replacement(range(u.shape[1]), deg):
            c = 1.0
            for i in set(idx):
                k = idx.count(i)
                c = c * he[k][:, i] / math.sqrt(math.factorial(k))
            cols.append(c)
    return np.stack(cols, 1)


def haar(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def rows(rng, d, J_s, J_x, sparse=None):
    """Signal and extra feature rows, with their Hermite degree."""
    if sparse:
        s, Q = sparse, np.zeros((J_s, d))
        perm = rng.permutation(d)
        for j in range(J_s):
            if s * J_s <= d:
                sup = perm[j * s:(j + 1) * s]
            else:
                sup = [j, *rng.choice(np.delete(np.arange(d), j), s - 1, replace=False)]
            Q[j, sup] = rng.standard_normal(s)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        E = rng.standard_normal((d, J_x))
        return np.vstack([Q, np.linalg.qr(E - Q.T @ (Q @ E))[0].T]), 1
    if J_s + J_x <= d:
        return haar(rng, d)[:J_s + J_x], 1
    sizes = np.cumsum([math.comb(d + k - 1, k) for k in range(1, 8)])
    K_s = sizes[np.searchsorted(sizes, J_s)]
    D = int(np.searchsorted(sizes, K_s + J_x)) + 1
    Q = np.zeros((J_s + J_x, sizes[D - 1]))
    Q[:J_s, :K_s] = haar(rng, K_s)[:J_s]
    Q[J_s:, K_s:K_s + J_x] = np.eye(J_x)
    return Q, D


def mstar(e, t):
    """True rank m*(t): active prefix of lam. Oscillating directions are excluded."""
    m = len(e["lam"])
    for t0, k in e["active"] or []:
        if t >= t0:
            m = k
    return m


def lam_t(e, t):
    """Signal spectrum lam(t): lam with active prefix and decay, then the oscillating directions."""
    lam = np.array(e["lam"], float)
    lam[mstar(e, t):] = 0
    if e["decay"]:
        lam *= np.exp(-np.array(e["decay"]) * t / e["T"])
    if e["osc"]:
        o = e["osc"]
        ph = 2 * np.pi * (t / o["period"] + np.arange(o["n"]) / o["n"])
        lam = np.r_[lam, o["base"] * (1 + o["amp"] * np.sin(ph))]
    return lam


def phi_t(e, t):
    """Rotation angle of the p side signal directions at round t."""
    r = e["rotation"]
    return r["rate"] * max(0, t - r["start"]) + (r["angle"] if r["at"] is not None and t >= r["at"] else 0.0)


def link(e, z, t, zn=0.0):
    """Apply the configured reward link and optional nuisance term."""
    if e["link"] == "id":
        return z
    if e["link"] == "tanh":
        return np.tanh(z)
    w = e["omega"][0] + e["omega"][1] * t / e["T"]
    h = lambda u: np.tanh(0.5 * u) + 2 * np.sin(w * u)
    f = h(e["c_z"] * z)
    return f + h(e["nuisance"]["scale"] * zn) if e["nuisance"] else f


def pmat(e, t):
    """Action factor map with signal rotation, spectrum, and spurious directions."""
    n = e["J"] + e["J_sp"]
    P = np.eye(n, e["Qp"].shape[0])
    if e["rotation"]:
        j, phi = np.arange(len(e["lam"])), phi_t(e, t)
        P[j, j], P[j, n + j] = np.cos(phi), np.sin(phi)
    return np.sqrt(np.r_[lam_t(e, t), e["lsp"]])[:, None] * P


def als(M, r, lam, iters, seed):
    """Rank r alternating ridge least squares U V' on the observed entries of M: (U, V)."""
    W = (~np.isnan(M)).astype(float)
    Y, I = np.nan_to_num(M), lam * np.eye(r)
    rng = np.random.default_rng(seed)
    U, V = 0.1 * rng.standard_normal((M.shape[0], r)), 0.1 * rng.standard_normal((M.shape[1], r))
    gram = lambda W, V: (W @ (V[:, :, None] * V[:, None]).reshape(len(V), -1)).reshape(-1, r, r)
    solve = lambda W, Y, V: np.linalg.solve(gram(W, V) + I, (Y @ V)[..., None])[..., 0]
    for _ in range(iters):
        U = solve(W, Y, V)
        V = solve(W.T, Y.T, U)
    return U, V


def user_pool(e, F, feat, X, rusers, reps, rpool):
    """Draw user rounds from a seeded pool and set reward noise."""
    T, n, N = e["T"], e["noise"], e.get("n_users")
    sig = np.linalg.svd(F, compute_uv=False)[:e["rank"]] / math.sqrt(F.size)

    sd = n["sd"] if n["sd"] is not None else n["rms"] * math.sqrt((F**2).mean()) if n.get("rms") else sig[-1] / n["rho"]
    pool = np.arange(len(F)) if N is None else rpool.permutation(len(F))[:N]
    users = pool[rusers.integers(len(pool), size=T + 1)]
    sp = np.linalg.svd(F[pool], compute_uv=False)[:min(len(pool), e["rank"])] / math.sqrt(F[pool].size)
    e.update(theta=feat[users], X=X, F=F, users=users, pool=pool, sigma=sig, sd=sd,
             eps=sd * reps.standard_normal(T + 1), lam=sp**2, active=None)
    return e


def kuairec_csv(name):
    """Load KuaiRec user, video, watch_ratio rows; download the archive if absent."""
    if not (DATA / "KuaiRec.zip").exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(KUAIREC, DATA / "KuaiRec.zip")
    with zipfile.ZipFile(DATA / "KuaiRec.zip").open(f"KuaiRec 2.0/data/{name}.csv") as f:
        return np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 7))


def kuairec(clip, d, ridge, iters):
    """Clipped small-matrix rewards and ALS features from the big matrix."""
    cache = DATA / f"kuairec_c{clip}_d{d}_r{ridge}_i{iters}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["W"], z["theta"], z["x"]
    S, R = kuairec_csv("small_matrix"), kuairec_csv("big_matrix")
    users, videos = np.unique(S[:, 0]).astype(int), np.unique(S[:, 1]).astype(int)
    W = np.full((len(users), len(videos)), np.nan)
    W[np.searchsorted(users, S[:, 0]), np.searchsorted(videos, S[:, 1])] = np.minimum(S[:, 2], clip)
    nv = int(R[:, 1].max()) + 1
    key, inv = np.unique(R[:, 0].astype(np.int64) * nv + R[:, 1].astype(np.int64), return_inverse=True)
    A = np.full((int(R[:, 0].max()) + 1, nv), np.nan)
    A.flat[key] = np.bincount(inv, np.minimum(R[:, 2], clip)) / np.bincount(inv)
    U, V = als(A, d, ridge, iters, 0)
    DATA.mkdir(exist_ok=True)
    np.savez(cache, W=W, theta=U[users], x=V[videos])
    return W, U[users], V[videos]


def kuairec_env(cfg, seed):
    """Centered KuaiRec rewards on seeded videos; missing values use video means."""
    e = {**cfg["env"], "T": cfg["T"]}
    assert e["d_theta"] == e["d_x"], "features of users and videos come from one ALS"
    rcols, rusers, reps, rpool = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(4))
    W, Uf, Vf = kuairec(e["clip"], e["d_x"], e["ridge"], e["iters"])
    cols = rcols.choice(W.shape[1], e["n_actions"], replace=False)
    selected, mean = W[:, cols], np.nanmean(W)
    F = np.where(np.isnan(selected), np.nanmean(W, 0)[cols], selected) - mean
    e.update(cols=cols, mean=mean)
    return user_pool(e, F, Uf, Vf[cols], rusers, reps, rpool)


def smooth_basis(rng, Z, ell, m):
    """QR of RBF GP draws, normalized for the uniform distribution on Z."""
    Kz = np.exp(-((Z[:, None] - Z[None]) ** 2).sum(-1) / (2 * ell**2))
    L = np.linalg.cholesky(Kz + 1e-8 * np.eye(len(Z)))
    return np.linalg.qr(L @ rng.standard_normal((len(Z), m)))[0] * math.sqrt(len(Z))


def smooth_env(cfg, seed):
    """Smooth rewards with the given spectrum and uniformly sampled contexts."""
    e, T = {**cfg["env"], "T": cfg["T"]}, cfg["T"]
    sig = np.array(e["sigma"], float)
    rth, rx, reps, rU, rV = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(5))
    pool, X = rth.standard_normal((e["n_ctx"], e["d_theta"])), rx.standard_normal((e["n_actions"], e["d_x"]))
    U, V = smooth_basis(rU, pool, e["ell_theta"], len(sig)), smooth_basis(rV, X, e["ell_x"], len(sig))
    ctx = rth.integers(e["n_ctx"], size=T + 1)
    e.update(pool=pool, ctx=ctx, theta=pool[ctx], X=X, U=U, V=V, F=((U * sig) @ V.T)[ctx],
             eps=e["noise_sd"] * reps.standard_normal(T + 1), lam=(sig**2).tolist(), active=None, sd=e["noise_sd"])
    return e


def make_env(cfg, seed):
    """Draw all environment randomness; rows 1..T are rounds, row 0 is initial state."""
    if cfg["env"].get("data") == "kuairec":
        return kuairec_env(cfg, seed)
    if cfg["env"].get("kind") == "smooth":
        return smooth_env(cfg, seed)
    e = {**cfg["env"], "T": cfg["T"]}
    T, K, dg, dp = cfg["T"], e["n_actions"], e["d_theta"], e["d_x"]
    rth, rx, reps, rmask, rrows = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(5))
    eta = rth.standard_normal((T + 1, dg))
    th = np.empty_like(eta)
    th[0] = SIG * eta[0]
    for t in range(1, T + 1):
        th[t] = RHO * th[t - 1] + SD_ETA * eta[t]
    if e["mask"] is not None:
        th = th * (rmask.random((T + 1, dg)) < e["mask"])
    n = e["noise"]
    eps = n["sd"] * reps.standard_normal(T + 1) if n["kind"] == "gauss" else n["scale"] * reps.standard_t(3, T + 1)
    X = rx.standard_normal((T + 1, K, dp))
    J = len(e["lam"]) + (e["osc"]["n"] if e["osc"] else 0)
    Jn = e["nuisance"]["J_n"] if e["nuisance"] else 0
    Qg, Dg = rows(rrows, dg, J, e["J_sp"] + Jn, e["sparse"])
    Qp, Dp = rows(rrows, dp, J, e["J_sp"] + (len(e["lam"]) if e["rotation"] else 0) + Jn, e["sparse"])
    e.update(theta=th, X=X, eps=eps, J=J, Jn=Jn, Qg=Qg, Dg=Dg, Qp=Qp, Dp=Dp,
             lsp=rrows.uniform(*e["lam_sp"], e["J_sp"]), a=hermite(th / SIG, Dg) @ Qg.T,
             b=(hermite(X.reshape(-1, dp), Dp) @ Qp.T).reshape(T + 1, K, -1))
    return e


def step(env, t):
    """Context, arm pool, and noiseless rewards at round t; noise is env['eps'][t]."""
    if env.get("data") == "kuairec":
        return env["theta"][t], env["X"], env["F"][env["users"][t]]
    if env.get("kind") == "smooth":
        return env["theta"][t], env["X"], env["F"][t]
    a, b, n, Jn = env["a"][t], env["b"][t], env["J"] + env["J_sp"], env["Jn"]
    z = b @ pmat(env, t).T @ a[:n]
    zn = b[:, -Jn:] @ a[-Jn:] / math.sqrt(Jn) if Jn else 0.0
    return env["theta"][t], env["X"][t], link(env, z, t, zn)
