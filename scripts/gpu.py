import copy
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from scripts.run import make_model, parameters


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def setup(device):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required.')
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def hardware(device):
    return dict(gpu=torch.cuda.get_device_name(device), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, scipy=__import__('scipy').__version__,
                cuda=torch.version.cuda, dtype='float64', cpu_threads=torch.get_num_threads())

def load_job(job):
    with np.load(INPUTS / job['stream'], allow_pickle=False) as data:
        stream = {key: data[key].copy() for key in data.files}
    if job['case'] == 'kuairec':
        stream = dict(theta=stream['contexts'][stream['indices']], rewards=stream['means'][stream['indices']],
                      X=stream['X'], noise=stream['noise'])
    with np.load(INPUTS / job['initial'], allow_pickle=False) as data:
        initial = {key: torch.from_numpy(data[key].copy()) for key in data.files}
    return stream, initial


def load_model(job, initial, device, method):
    plan = dict(paper_key='physical.' + job['case'],
                environment='KuaiRec' if job['case'] == 'kuairec' else job['case'].capitalize(),
                settings=dict(fold_round=300))
    model = make_model(plan, copy.deepcopy(job['config']), method, job['seed'], device=device)
    model.encoder.net.load_state_dict({key: initial[key] for key in ('0.weight', '0.bias', '2.weight', '2.bias')})
    with torch.no_grad():
        model.B.copy_(initial['B'])
        model.hyp.copy_(initial['hyp'])
    model.rng.bit_generator.state = copy.deepcopy(job['fit_rng'])
    return model


def counts(model):
    return dict(parameters(model), neural_output_width=model.m_max, gp_dimension=model.m)


def warmup(job, stream, initial, device):
    model = load_model(job, initial, device, 'BBP')
    X, n = stream['X'][:4], 40
    ids = np.arange(n) % len(X)
    model.thetas = list(torch.as_tensor(stream['theta'][:n], dtype=torch.float64, device=device))
    model.Xs = list(torch.as_tensor(X[ids], dtype=torch.float64, device=device))
    model.ys = list(torch.as_tensor(stream['rewards'][np.arange(n), ids] + stream['noise'][:n],
                                   dtype=torch.float64, device=device))
    model.D2 = torch.cdist(model.x, model.x).square()
    query = model.th[-1].cpu().numpy() + .137
    for width in (100, 2):
        model._set_projection(np.eye(100)[:, :width], width)
        actual, reference = model.mean_var(query, X), model._dual_posterior(X, query)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a.cpu(), b, atol=1e-7, rtol=1e-7)
    model._fold()
    model._fit_mll(steps=10)
    del model
    gc.collect()
    synchronize(device)


def fresh_calibration(job, stream, t):
    rng = np.random.default_rng([job['seed'], 90211, t])
    n, d = (max(128, t), stream['theta'].shape[1])
    raw = rng.normal(size=(n, d))
    rho = job.get('calibration_spec', {}).get('nuisance_correlation', 0.0)
    if rho:
        raw[:, 3:] = np.sqrt(1 - rho) * raw[:, 3:] + np.sqrt(rho) * rng.normal(size=(n, 1))
    return raw @ stream['rotation']

def worker(index, method, device):
    job = json.loads((INPUTS / 'protocol.json').read_text())['jobs'][index]
    stream, initial = load_job(job)
    record = dict(case=job['case'], seed=job['seed'], method=method,
                  horizon=job['horizon'], config=job['config'], trajectory=[], hardware=hardware(device))
    warmup(job, stream, initial, device)
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    start, cpu = time.perf_counter(), time.process_time()
    model = load_model(job, initial, device, method)
    initial_counts = counts(model)
    boundary = None
    for i in range(job['horizon']):
        t = i + 1
        theta, rewards, X = stream['theta'][i], stream['rewards'][i], stream['X']
        chosen, _, _ = model.select_action(X, theta)
        action = int(np.argmin(np.linalg.norm(X - chosen, axis=1)))
        reward = float(rewards[action] + stream['noise'][i])
        record['trajectory'].append(dict(t=t, action=action, reward=reward,
            regret=float(rewards.max() - rewards[action]), rank=model.m,
            nn_width=model.m_max, epsilon=model.epsilon, theta=theta.tolist(), chosen_x=X[action].tolist()))
        calibration = None
        if job['case'] != 'kuairec' and t >= 100 and t % 10 == 0 and (method == 'BBP' or t <= 300):
            calibration = fresh_calibration(job, stream, t)
        model.update(theta, X[action], reward, calibration_contexts=calibration)
        if t == 300:
            synchronize(device)
            boundary = time.perf_counter()
    synchronize(device)
    stop, cpu_stop = time.perf_counter(), time.process_time()
    record['timing'] = dict(wall_seconds=stop-start, cpu_seconds=cpu_stop-cpu,
                            post300_wall_seconds=stop-boundary)
    rows = record['trajectory']
    record['summary'] = dict(regret=sum(r['regret'] for r in rows),
        regret_post100=sum(r['regret'] for r in rows[100:]),
        regret_post300=sum(r['regret'] for r in rows[300:]),
        mean_gp_dimension_post100=statistics.mean(r['rank'] for r in rows[100:]))
    record['parameters'] = dict(initial=initial_counts, final=counts(model))
    record['spectra'], record['transfer'] = model.spectral_history, model.transfer
    suffix = 'BBP' if method == 'BBP' else 'compressed'
    save(OUTPUT / 'runs' / f"{job['case']}_{job['seed']}_{suffix}.json", record)


if __name__ == '__main__':
    INPUTS, OUTPUT = Path(sys.argv[1]), Path(sys.argv[2])
    device = torch.device('cuda:0')
    setup(device)
    worker(int(sys.argv[3]), sys.argv[4], device)
