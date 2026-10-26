"""Crossed-grid GP training and frozen-feature BBP sampling."""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from bandits.nn_agp_adaptive import NN_AGP_UCB_Adaptive

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
JITTER = 1e-6
CLAMP = {"ell": (math.log(.05), math.log(20.)), "s2": (math.log(1e-3), math.log(1e2))}
M, C, HIDDEN, A, ACTION_BLOCK = 128, 512, 1024, 262144, 4096
SEED, SD, NOISE, RHO = 96015, .02, .005, .005
RADIUS = math.sqrt(-math.log(RHO))
STAGE1, STAGE2 = 2000, 8000
FIRST_MC, CALIBRATION_MC, MC_BATCH, FIT_MC = 262144, 1000000, 4096, 32768
CONFIG = dict(T=C * A, env=dict(d_theta=M, d_x=A), agent=dict(
    M=M, hidden=[HIDDEN, HIDDEN], noise_sd=NOISE, cov_lr=.001,
    cov_mle_max=C * A, cov_steps=1, T_spec=10, eps=.01))
GATES = dict(mean_energy_share=.01, tail_cv=.10, heldout_affine_prediction_error=.02,
             residual_max_eigenvalue_over_background_variance=.02,
             teacher_top2_r2=.8, leading_kurtosis_distance_from_3=.3)
REPS, CHUNK, SAMPLE_SEED = 400, 10, 97013


def f64(value):
    return torch.as_tensor(value, dtype=torch.float64)


def dense_nll(K, y, noise):
    L = torch.linalg.cholesky(K + (noise + JITTER) * torch.eye(len(K), dtype=K.dtype))
    return .5 * y @ torch.cholesky_solve(y[:, None], L)[:, 0] + L.diagonal().log().sum()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(ROOT / name) for name in ("paper_bbp.py", "bandits/nn_agp_adaptive.py")}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def merge_stats(n, mean, S, block):
    """Parallel Welford update along the action axis, preserving raw rewards."""
    count = block.shape[1]
    block_mean = block.mean(1)
    centered = block - block_mean[:, None]
    block_S = centered @ centered.T
    if n == 0:
        return count, block_mean, block_S
    delta = block_mean - mean
    new_n = n + count
    return new_n, mean + delta * (count / new_n), S + block_S + np.outer(delta, delta) * (n * count / new_n)


def nll(agent, theta, z, S, actions):
    H = agent.features(theta)
    G = agent.hyp[1].exp() * (H @ H.T)
    exponent = -RADIUS ** 2 / agent.hyp[0].exp().square()
    rho, perpendicular = exponent.exp(), -torch.expm1(exponent)
    eta = agent.hyp[2].exp() + JITTER
    eye = torch.eye(len(theta), dtype=torch.float64)
    L0 = torch.linalg.cholesky(eta * eye + perpendicular * G)
    L1 = torch.linalg.cholesky(eta * eye + (1 + (actions - 1) * rho) * G)
    return ((actions - 1) * L0.diagonal().log().sum() + L1.diagonal().log().sum()
            + .5 * torch.trace(torch.cholesky_solve(S, L0))
            + .5 * z @ torch.cholesky_solve(z[:, None], L1)[:, 0])


def verify():
    rng = np.random.default_rng(SEED)
    theta, Y = rng.standard_normal((4, 3)), rng.standard_normal((4, 11))
    count, mean, S = 0, np.zeros(4), np.zeros((4, 4))
    for block in (Y[:, :3], Y[:, 3:7], Y[:, 7:]):
        count, mean, S = merge_stats(count, mean, S, block)
    centered = Y - Y.mean(1, keepdims=True)
    stats_error = float(np.max(np.abs(S - centered @ centered.T)))
    cfg = dict(env=dict(d_theta=3, d_x=11), agent=dict(M=3, hidden=[8, 8], noise_sd=.07, T_spec=10))
    agent = NN_AGP_UCB_Adaptive.from_config(cfg, SEED, "Full")
    th, y, z, s = f64(theta), f64(Y), f64(mean * math.sqrt(count)), f64(S)
    X = RADIUS * torch.eye(count, dtype=torch.float64)
    full_th, full_x = th.repeat_interleave(count, dim=0), X.repeat(len(th), 1)
    H = agent.features(full_th)
    parameters = [*agent.encoder.net.parameters(), agent.B, agent.hyp]
    dense = dense_nll(agent.kernel(H, H, torch.cdist(full_x, full_x).square()), y.flatten(), agent.hyp[2].exp())
    gd = torch.autograd.grad(dense, parameters)
    reduced = nll(agent, th, z, s, count)
    gr = torch.autograd.grad(reduced, parameters)
    grad_error = max(float((u - v).abs().max()) for u, v in zip(gd, gr))
    loss_error = float(abs(dense - reduced).detach())
    result = dict(Welford_centered_S_max_absolute_error=stats_error,
                  dense_nll_absolute_error=loss_error, all_parameter_gradient_max_absolute_error=grad_error)
    assert stats_error < 1e-10 and loss_error < 1e-7 and grad_error < 1e-6, result
    return result


def rewards(output):
    start = time.monotonic()
    rng = np.random.default_rng(SEED)
    random = SD * rng.standard_normal((M, M))
    theta = np.r_[SD * math.sqrt(M) * np.eye(M), -SD * math.sqrt(M) * np.eye(M), random, -random]
    U, _ = np.linalg.qr(rng.standard_normal((M, M)))
    q = np.r_[2., 1.5, np.ones(M - 2)] / M
    common_u = rng.standard_normal((1, M))
    rng_before_blocks = rng.bit_generator.state
    truth_map = 2 * (theta @ U * np.sqrt(q))
    count, mean, S = 0, np.zeros(C), np.zeros((C, C))
    for offset in range(0, A, ACTION_BLOCK):
        size = min(ACTION_BLOCK, A - offset)
        Z = rng.standard_normal((size, M))
        P = math.sqrt(1 - RHO) * Z + math.sqrt(RHO) * common_u
        Yblock = truth_map @ P.T + NOISE * rng.standard_normal((C, size))
        count, mean, S = merge_stats(count, mean, S, Yblock)
        if count % (ACTION_BLOCK * 16) == 0:
            print(json.dumps(dict(streamed_actions=count, seconds=time.monotonic() - start)), flush=True)
    S = (S + S.T) / 2
    z = mean * math.sqrt(count)
    np.savez_compressed(output / "training_statistics.npz", theta=theta, U=U, q=q,
                        common_u=common_u, z=z, centered_action_S=S, action_mean=mean,
                        actions=A, action_block=ACTION_BLOCK, seed=SEED)
    return theta, U, z, S, dict(seconds=time.monotonic() - start, actions=count,
        rng_state_before_action_blocks=rng_before_blocks,
        centered_S_min_eigenvalue=float(np.linalg.eigvalsh(S).min()))


def save_checkpoint(agent, stage, steps, status, output):
    torch.save(dict(g=agent.encoder.net.state_dict(), B=agent.B.detach(), hyp=agent.hyp.detach(),
                    config=CONFIG, seed=SEED, actions=A, stage=stage,
                    completed_steps=steps, status=status, optimizer=agent.optimizer.state_dict()), output / "checkpoint.pt")


def geometry(agent, U, contexts, seed_suffix, label, output):
    rng = np.random.default_rng([SEED, seed_suffix])
    count, sum_H, sum_HH = 0, np.zeros(M), np.zeros((M, M))
    train_theta, train_H, coeff = [], [], None
    residual_sum, residual_outer = np.zeros(M), np.zeros((M, M))
    raw_energy, residual_energy, heldout_n = 0., 0., 0
    start = time.monotonic()
    for offset in range(0, contexts, MC_BATCH):
        theta = SD * rng.standard_normal((min(MC_BATCH, contexts - offset), M))
        with torch.no_grad():
            H = agent.features(f64(theta)).numpy()
        count += len(theta)
        sum_H += H.sum(0)
        sum_HH += H.T @ H
        if offset < FIT_MC:
            train_theta.append(theta)
            train_H.append(H)
            if offset + len(theta) == FIT_MC:
                fit_theta, fit_H = np.concatenate(train_theta), np.concatenate(train_H)
                coeff = np.linalg.lstsq(np.c_[np.ones(FIT_MC), fit_theta], fit_H, rcond=None)[0]
        else:
            residual = H - np.c_[np.ones(len(theta)), theta] @ coeff
            residual_sum += residual.sum(0)
            residual_outer += residual.T @ residual
            residual_energy += np.sum(residual ** 2)
            raw_energy += np.sum(H ** 2)
            heldout_n += len(theta)
        if count % 262144 == 0:
            print(json.dumps(dict(geometry=label, contexts=count, seconds=time.monotonic() - start)), flush=True)
    mean, J = sum_H / count, sum_HH / count
    Cov = J - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(J)
    values, vectors = values[::-1], vectors[:, ::-1]
    tail = values[2:]
    residual_mean = residual_sum / heldout_n
    residual_cov = residual_outer / heldout_n - np.outer(residual_mean, residual_mean)
    residual_values = np.linalg.eigvalsh(residual_cov)
    fitting_scores = fit_H @ vectors[:, :2]
    signal_coef = np.linalg.lstsq(np.c_[np.ones(FIT_MC), fitting_scores], fit_theta @ U[:, :2], rcond=None)[0]
    rng = np.random.default_rng([SEED, seed_suffix])
    moments = np.zeros((4, M))
    target_sum, target_square_sum, target_error = np.zeros(2), np.zeros(2), np.zeros(2)
    for offset in range(0, contexts, MC_BATCH):
        theta = SD * rng.standard_normal((min(MC_BATCH, contexts - offset), M))
        with torch.no_grad():
            H = agent.features(f64(theta)).numpy()
        scores = H @ vectors
        for power in range(1, 5):
            moments[power - 1] += (scores ** power).sum(0)
        if offset >= FIT_MC:
            targets = theta @ U[:, :2]
            predictions = np.c_[np.ones(len(theta)), scores[:, :2]] @ signal_coef
            target_sum += targets.sum(0)
            target_square_sum += (targets ** 2).sum(0)
            target_error += ((targets - predictions) ** 2).sum(0)
    moments /= count
    variance = moments[1] - moments[0] ** 2
    fourth = moments[3] - 4 * moments[0] * moments[2] + 6 * moments[0] ** 2 * moments[1] - 3 * moments[0] ** 4
    kurtosis = fourth / np.maximum(variance ** 2, 1e-30)
    r2 = 1 - target_error / (target_square_sum - target_sum ** 2 / heldout_n)
    quality = dict(method="Batched Monte Carlo of unchanged raw features; uncentered J", contexts=count,
        context_sd=SD, seed=[SEED, seed_suffix], affine_fit_contexts=FIT_MC, heldout_contexts=heldout_n,
        mean_energy=float(mean @ mean), mean_energy_share=float(mean @ mean / np.trace(J)),
        background_sigma_squared=float(tail.mean()),
        mean_energy_over_background_sigma_squared=float(mean @ mean / tail.mean()),
        tail_cv=float(tail.std() / tail.mean()), tail_participation=float(tail.sum() ** 2 / (tail @ tail)),
        tail_min_over_mean=float(tail.min() / tail.mean()), tail_max_over_mean=float(tail.max() / tail.mean()),
        leading_values=values[:4].tolist(), heldout_affine_prediction_error=float(residual_energy / raw_energy),
        heldout_affine_residual_covariance_max_eigenvalue=float(residual_values[-1]),
        heldout_affine_residual_max_eigenvalue_over_background_sigma_squared=float(residual_values[-1] / tail.mean()),
        teacher_top2_heldout_r2=r2.tolist(), score_kurtosis=kurtosis.tolist(),
        learned_hyp=agent.hyp.detach().exp().tolist(), seconds=time.monotonic() - start)
    quality["gates"] = dict(mean=quality["mean_energy_share"] < GATES["mean_energy_share"],
        background=quality["tail_cv"] < GATES["tail_cv"],
        linearity=quality["heldout_affine_prediction_error"] < GATES["heldout_affine_prediction_error"],
        residual_covariance=quality["heldout_affine_residual_max_eigenvalue_over_background_sigma_squared"] < GATES["residual_max_eigenvalue_over_background_variance"],
        useful_signals=bool(min(r2) > GATES["teacher_top2_r2"]),
        leading_gaussian_moments=bool(np.max(np.abs(kurtosis[:2] - 3)) < GATES["leading_kurtosis_distance_from_3"]))
    quality["pass"] = all(quality["gates"].values())
    np.savez_compressed(output / f"{label}.npz", mean=mean, C=Cov, J=J, values=values, vectors=vectors,
        affine_coefficient=coeff, signal_coefficient=signal_coef, teacher_rotation=U,
        heldout_affine_residual_C=residual_cov, heldout_affine_residual_values=residual_values)
    write_json(output / f"{label}.json", quality)
    print(json.dumps(dict(geometry=label, quality=quality)), flush=True)
    return quality


def train(output):
    """Run all 2,000 + 8,000 updates and gated calibration in output."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    hashes = source_hashes()
    verification = verify()
    protocol = dict(config=CONFIG, seed=SEED, M=M, C=C, A=A, hidden=HIDDEN,
        context_sd=SD, observation_noise_sd=NOISE, teacher_rho=RHO, action_radius=RADIUS,
        budgets=dict(stage1_steps=STAGE1, stage2_steps=STAGE2,
                     first_geometry_contexts=FIRST_MC, independent_calibration_contexts=CALIBRATION_MC),
        verification=verification, source_hashes=hashes, geometry_gates=GATES,
        context_algorithm="Paired SD*sqrt(M) coordinate basis, then M IID Gaussian context draws and their negatives",
        rng_algorithm="NumPy default_rng(seed96015): Gaussian context MxM; QR Gaussian MxM teacher U; one common_u 1xM; then action blocks",
        action_block_algorithm="Block size4096: draw independent Z[block,M], P=sqrt(1-rho)*Z+sqrt(rho)*common_u; then Gaussian noise[C,block]",
        genuine_actions="x_a=R e_a in R^A, implicit compound RBF covariance (1-rho)I+rho11'",
        statistics="Parallel Welford action means and centered scatter S; z=sqrt(A)*mean",
        teacher_q=(np.r_[2., 1.5, np.ones(M - 2)] / M).tolist(),
        optimization="Default random NN/B/hyp initialization; exact NLL/A; Adam lr.001 2000steps, Adam reset cosine1e-4 to1e-6 8000steps; clip5/clamps",
        geometry="First262144 fresh Gaussian raw contexts; affine fit32768; if all gates pass, independent seed calibration1m",
        feature_integrity="Unchanged raw h=g(theta)B; final checkpoint; no best selection")
    write_json(output / "training_protocol.json", protocol)
    theta, U, z, S, streaming = rewards(output)
    protocol["streaming"] = streaming
    write_json(output / "training_protocol.json", protocol)
    agent = NN_AGP_UCB_Adaptive.from_config(CONFIG, SEED, "Full")
    initial_g = {key: value.detach().clone() for key, value in agent.encoder.net.state_dict().items()}
    initial_B, initial_hyp = agent.B.detach().clone(), agent.hyp.detach().clone()
    th, z, S = f64(theta), f64(z), f64(S)
    with torch.no_grad():
        initial_loss = float(nll(agent, th, z, S, A) / A)
    trace, completed, train_start = [], {"stage1": 0, "stage2": 0}, time.monotonic()
    last_progress = train_start
    for stage, total in ((1, STAGE1), (2, STAGE2)):
        if stage == 2:
            agent.optimizer = torch.optim.Adam([*agent.encoder.net.parameters(), agent.B, agent.hyp], lr=1e-4)
        for step in range(1, total + 1):
            lr = (.001 if stage == 1 else
                  1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * (step - 1) / (total - 1))))
            agent.optimizer.param_groups[0]["lr"] = lr
            agent.optimizer.zero_grad()
            loss = nll(agent, th, z, S, A) / A
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(agent.optimizer.param_groups[0]["params"], 5.)
            assert torch.isfinite(loss) and torch.isfinite(norm), (stage, step, loss, norm)
            agent.optimizer.step()
            with torch.no_grad():
                agent.hyp[0].clamp_(*CLAMP["ell"])
                agent.hyp[1].clamp_(*CLAMP["s2"])
                agent.hyp[2].clamp_(math.log(1e-6), math.log(100.))
            completed[f"stage{stage}"] = step
            if step == 1 or step % 500 == 0 or time.monotonic() - last_progress > 30:
                row = dict(stage=stage, step=step, lr=lr, loss=float(loss.detach()),
                    gradient_norm_before_clip=float(norm), training_seconds=time.monotonic() - train_start,
                    hyp=agent.hyp.detach().exp().tolist())
                trace.append(row)
                print(json.dumps(row), flush=True)
                write_json(output / "progress.json", dict(initial_loss=initial_loss, completed_steps=completed, trace=trace))
                last_progress = time.monotonic()
        save_checkpoint(agent, stage, completed, "completed", output)
    training_seconds = time.monotonic() - train_start
    with torch.no_grad():
        final_loss = float(nll(agent, th, z, S, A) / A)
    updates = {key: float((value - initial_g[key]).abs().max()) for key, value in agent.encoder.net.state_dict().items()}
    updates.update(B=float((agent.B.detach() - initial_B).abs().max()),
                   hyp=float((agent.hyp.detach() - initial_hyp).abs().max()))
    result = dict(status="completed", completed_steps=completed, initial_loss=initial_loss,
        final_loss=final_loss, training_seconds=training_seconds, streaming=streaming,
        trace=trace, parameter_updates_from_random_initialization=updates,
        source_unchanged=hashes == source_hashes())
    write_json(output / "training_results.json", result)
    first = geometry(agent, U, FIRST_MC, 42, "geometry_first", output)
    result["first_geometry"] = first
    write_json(output / "training_results.json", result)
    if first["pass"]:
        calibration = geometry(agent, U, CALIBRATION_MC, 43, "calibration", output)
        result["independent_calibration"] = calibration
        result["geometry_pass"] = calibration["pass"]
    else:
        result["geometry_pass"] = False
        result["calibration_skipped"] = "Initial raw geometry failed predeclared gates"
    result["total_seconds"] = time.monotonic() - start
    write_json(output / "training_results.json", result)
    print(json.dumps(result), flush=True)
    return result


def summary(values):
    mean = values.mean()
    se = values.std(ddof=1)/np.sqrt(len(values))
    return dict(mean=float(mean), se=float(se), ci80=[float(mean-1.281551565545*se),
                                                    float(mean+1.281551565545*se)])


def measure(H, n, targets):
    values, vectors = np.linalg.eigh(np.einsum("rni,rnj->rij", H[:, :n], H[:, :n])/n)
    block = np.einsum("rmi,mj->rij", vectors[:, :, -2:][:, :, ::-1], targets)**2
    return np.c_[values[:, -1], values[:, -2], block[:, 0, 0], block[:, 1, 1],
                 block.sum((1, 2))/2]


def sample(output):
    """Read checkpoint/calibration; save 400 repetitions in output/classical_transition."""
    output = Path(output)
    out = output / "classical_transition"
    out.mkdir(exist_ok=True)
    state = torch.load(output/"checkpoint.pt", weights_only=False)
    agent = NN_AGP_UCB_Adaptive.from_config(state["config"], state["seed"], "Full")
    agent.encoder.net.load_state_dict(state["g"])
    with torch.no_grad():
        agent.B.copy_(state["B"])
        agent.hyp.copy_(state["hyp"])
    calibration = np.load(output/"calibration.npz")
    mean, J = calibration["mean"], calibration["J"]
    values, V = np.linalg.eigh(J)
    values, V = values[::-1], V[:, ::-1]
    M = len(values)
    background = float(values[2:].mean())
    delta = values[:2]/background-1
    assert background>0 and delta[0]>delta[1]>0
    critical = M/delta**2
    sizes = sorted(set([max(2, round(n*f)) for n in critical
                        for f in (.125, .25, .5, .75, 1, 1.5, 2, 4, 8, 16)]))
    assert max(sizes)<=32768
    sd = .02
    targets = V[:, :2]
    C = J-np.outer(mean, mean)
    eig, Q = np.linalg.eigh(C)
    measured_root = Q*np.sqrt(np.maximum(eig, 0))
    isotropic_values = np.r_[values[:2], np.full(M-2, background)]
    classical_root = V*np.sqrt(isotropic_values)
    inputs = {"checkpoint.pt": output/"checkpoint.pt", "calibration.npz": output/"calibration.npz",
              "paper_bbp.py": ROOT/"paper_bbp.py", "bandits/nn_agp_adaptive.py": ROOT/"bandits/nn_agp_adaptive.py"}
    hashes = {name: digest(path) for name, path in inputs.items()}
    protocol = dict(repetitions=REPS, seed=SAMPLE_SEED, sizes=sizes,
        context_sd=sd, M=M, sigma_eff2=background, spike_excess=(values[:2]-background).tolist(),
        spike_ratio=delta.tolist(), ncrit=critical.tolist(), input_hashes=hashes,
        statistic="Unmodified trained h=g(theta)B; raw H.T@H/n on fresh IID contexts",
        calibration=("Exact Gaussian-input ReLU moments" if (output/"exact_calibration.json").exists()
                     else "Independent large Monte Carlo calibration")+"; fixed raw J and directions before fresh tests",
        controls="Finite Gaussian with isotropic tail; finite Gaussian with actual mean and covariance. Diagnostic only",
        scope="Frozen learned checkpoint after offline crossed GP likelihood training; n counts independent contexts, not rewards",
        phase_labels="Asymptotic BBP reference regimes; finite-M transition is rounded",
        intervals="80% Monte Carlo confidence for means, conditional on this learned checkpoint")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    data = {kind: {n: [] for n in sizes} for kind in ("actual", "classical_Gaussian", "measured_Gaussian")}
    chunk = min(CHUNK, max(1, 256//M))
    for start in range(0, REPS, chunk):
        count = min(chunk, REPS-start)
        rng = np.random.default_rng([SAMPLE_SEED, start])
        z = rng.standard_normal((count, max(sizes), M))
        with torch.no_grad():
            flat = z.reshape(-1, M)
            H = np.concatenate([agent.features(f64(sd*flat[i:i+4096])).numpy()
                                for i in range(0, len(flat), 4096)]).reshape(z.shape)
        control_z = rng.standard_normal(z.shape)
        classic = control_z @ classical_root.T
        measured = mean+control_z @ measured_root.T
        for n in sizes:
            for name, features in (("actual", H), ("classical_Gaussian", classic), ("measured_Gaussian", measured)):
                data[name][n].append(measure(features, n, targets))
        if (start+count)%100==0:
            print(f"Fresh repetitions {start+count}/{REPS}", flush=True)
    rows, arrays = [], {}
    keys = ("peak1", "peak2", "overlap1", "overlap2", "projector")
    for n in sizes:
        gamma = M/n
        edge = background*(1+np.sqrt(gamma))**2
        outliers = [float(background*(1+d)*(1+gamma/d)) if d>np.sqrt(gamma) else edge for d in delta]
        overlap = [float((1-gamma/d**2)/(1+gamma/d)) if d>np.sqrt(gamma) else 0. for d in delta]
        row = dict(n=n, edge=edge, predicted_peaks=outliers, predicted_overlap=overlap)
        for kind in data:
            sample = np.concatenate(data[kind][n])
            arrays[f"{kind}_n{n}"] = sample
            row[kind] = {key: summary(sample[:, j]) for j, key in enumerate(keys)}
        rows.append(row)
    np.savez_compressed(out/"samples.npz", **arrays)
    result = dict(protocol=protocol, observations=rows,
                  inputs_unchanged=all(digest(inputs[name])==sha for name,sha in hashes.items()))
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(ncrit=critical.tolist(), inputs_unchanged=result["inputs_unchanged"], last=rows[-1])), flush=True)
    return result
