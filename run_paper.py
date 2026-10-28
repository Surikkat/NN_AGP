import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'paper_experiments'
BASELINES = ('Fixed', 'GP mult', 'GP add', 'NeuralUCB', 'NN-UCB')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = list((ROOT/'bandits').glob('*.py')) + [ROOT/name for name in
        ('run_paper.py', 'plot_paper.py', 'paper_envs.py', 'paper_bbp.py', 'requirements-paper.txt')]
    paths += list((ROOT/'datasets').glob('kuairec_c5.0_d20_r5.0_i15.npz'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def protocols():
    return {p.parent.name: read(p) for p in sorted(DATA.glob('*/EXP*/protocol.json'))}


def configuration(plan, record):
    cfg = copy.deepcopy(plan['configurations'][record['configuration']])
    if 'env' not in cfg:
        reference = next(r for r in plan['runs'] if r['method']=='Full' and r['seed']==record['seed'])
        cfg = copy.deepcopy(plan['configurations'][reference['configuration']])
    cfg.setdefault('T', plan['horizon'])
    return cfg


def environment(plan, cfg, seed):
    import numpy as np
    import paper_envs
    fresh = plan['environment'] in ('Weak', 'Balanced')
    if fresh:
        spec = plan['settings'].get('reward_specification') or dict(
            baseline=12., weak=.25 if plan['environment']=='Weak' else 1., noise=.25,
            d_theta=16, nuisance_correlation=0., actions=16)
        env = paper_envs.Environment(spec, seed, plan['horizon'])
    else:
        settings = copy.deepcopy(cfg)
        nuisance = plan['paper_key']=='online.rank3_d128'
        if nuisance:
            settings['env']['d_theta'] = 5
        env = paper_envs.make_env(settings, seed)
        if nuisance:
            extra = np.random.default_rng([seed,77091]).standard_normal((cfg['T']+1,123))
            env['theta'] = np.column_stack((env['theta'], extra))
            env['d_theta'] = 128
    return env, fresh


def make_model(plan, cfg, method, seed):
    import numpy as np
    import torch
    if method not in BASELINES:
        from bandits.nn_agp_adaptive import NN_AGP_UCB_Adaptive
        physical = plan['paper_key'].startswith('physical_fold300.')
        return NN_AGP_UCB_Adaptive.from_config(cfg, seed, 'BBP' if method=='Late' else method,
            calibration='fresh' if plan['environment'] in ('Weak','Balanced') else 'history',
            legacy=plan['paper_key'].startswith('legacy.'),
            corrected_svd=plan['settings'].get('corrected_svd_policy',True),
            fold_round=300 if physical and method=='Late' else None, catalog=physical)
    d, x = cfg['env']['d_theta'], cfg['env']['d_x']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if method=='Fixed':
        from bandits.nn_agp import NN_AGP_UCB
        return NN_AGP_UCB(d,x,m=math.ceil(d/3+x/10)+3,hidden_dim=64,lr=1e-3,
            mll_steps=40,beta=2.,init_noise=cfg['agent']['noise_sd'],dtype=torch.float64,batched=True)
    if method.startswith('GP '):
        from bandits.gp_ucb_multikernel import GP_UCB_KernelFlexible
        return GP_UCB_KernelFlexible(x,d,cfg['agent']['noise_sd'],beta=2.,
            kernel_type='multiplicative' if method=='GP mult' else 'adaptive',
            adaptive_weights={'theta':.6,'x':.4})
    if method=='NeuralUCB':
        from bandits.neural_ucb import NeuralUCB
        return NeuralUCB(input_dim=d+x,T=plan['horizon'],lambd=1.,nu=1.,delta=.01,
            S=1.,eta=1e-3,J=20,m=64,L=2,dual=True)
    from bandits.nn_ucb import NN_UCB
    return NN_UCB(context_dim=d+x,hidden_width=256,hidden_layers=2,lambda_=1.,
        beta=1.,J=10,lr=1e-3,seed=seed,device='cpu')


def reference_rows(raw, method):
    if 'r' in raw:
        return [dict(t=i+1,action=a,regret=r,rank=m,reward=y) for i,(a,r,m,y) in
            enumerate(zip(raw['a'],raw['r'],raw['mhat'],raw['y']))]
    rows = raw['trajectory']
    if isinstance(rows,dict):
        return rows[method]
    return rows if 'method' in raw else [r[method] for r in rows]


def parameters(model):
    if not hasattr(model,'hyp'):
        return None
    neural = sum(p.numel() for p in model.encoder.parameters())
    return dict(neural=neural,B=model.B.numel(),hyperparameters=model.hyp.numel(),
        total=neural+model.B.numel()+model.hyp.numel())


def checkpoint(model):
    result = {}
    for key in ('encoder','net','model','optimizer'):
        value = getattr(model,key,None)
        if value is not None and hasattr(value,'state_dict'):
            result[key] = value.state_dict()
    for key in ('B','hyp','projection','log_lengthscale','log_variance','log_noise',
        'thetas','Xs','ys','D2','m','m_max','step_count','history_X','history_y',
        'theta0_vec','A_diag','X','y','spectral_history','numerical_fallbacks'):
        if hasattr(model,key):
            result[key] = getattr(model,key)
    if hasattr(model,'rng'):
        result['fit_rng'] = model.rng.bit_generator.state
    if hasattr(model,'Z') and hasattr(model.Z,'factors'):
        result.update(gram_diagonal=model.Z.diagonal,gram_factors=model.Z.factors)
    return result


def train(plan, record):
    import numpy as np
    import torch
    import paper_envs
    import resource
    cfg = configuration(plan,record)
    method, seed = record['method'],record['seed']
    env, fresh = environment(plan,cfg,seed)
    cpu,wall = time.process_time(),time.perf_counter()
    model = make_model(plan,cfg,method,seed)
    adaptive = method not in BASELINES
    initial,rows = parameters(model),[]
    for t in range(1,plan['horizon']+1):
        if fresh:
            theta,X,rewards,noise = env.theta[t-1],env.X,env.rewards[t-1],float(env.noise[t-1])
        else:
            theta,X,rewards = paper_envs.step(env,t)
            noise = float(env['eps'][t])
        if t==1 and method in ('GP mult','GP add'):
            order = [m for m in np.roll(BASELINES,(1+seed)%len(BASELINES)) if m.startswith('GP ')]
            if order.index(method):
                np.random.randint(len(X))  # Preserve the paired baselines' first-round RNG order.
        if method in ('NeuralUCB','NN-UCB'):
            inputs = np.hstack((np.tile(theta,(len(X),1)),X))
            selected = model.select_action(inputs)
            action = int(selected[2] if method=='NeuralUCB' else selected)
        else:
            chosen,_,_ = model.select_action(X,theta)
            action = int(np.argmin(np.linalg.norm(X-chosen,axis=1)))
        reward = float(rewards[action]+noise)
        rank = model.m if adaptive or method=='Fixed' else None
        row = dict(t=t,action=action,theta=theta.tolist(),chosen_x=X[action].tolist(),
            reward=reward,regret=float(rewards.max()-rewards[action]),rank=rank,
            nn_width=getattr(model,'m_max',None),epsilon=getattr(model,'epsilon',None))
        digest = hashlib.sha256()
        for value in (theta,X,rewards,np.asarray([noise])):
            digest.update(str(value.shape).encode());digest.update(value.tobytes())
        row['pool_sha256'] = digest.hexdigest()
        if method=='NeuralUCB':
            model.update(None,np.hstack((theta,X[action])),reward)
        elif method=='NN-UCB':
            model.update(np.hstack((theta,X[action])),action,reward)
        elif method in BASELINES:
            model.update(theta,X[action],reward)
        else:
            calibration = None
            if fresh and t>=100 and t%cfg['agent']['T_spec']==0 and (method!='Late' or t<=300):
                calibration = env.calibration(seed,t,max(128,t))
            model.update(theta,X[action],reward,calibration_contexts=calibration)
        row['rank_after'] = model.m if adaptive else rank
        rows.append(row)
    timing = dict(cpu=time.process_time()-cpu,wall=time.perf_counter()-wall)
    old = reference_rows(read(DATA/record['result_file']),record['raw_method'])
    differences = {name:[r['t'] for r,s in zip(rows,old) if name in s and
        (r[name]!=s.get(name) if name in ('action','rank') else not math.isclose(r[name],s[name],rel_tol=1e-9,abs_tol=1e-9))]
        for name in ('action','rank','regret','reward')}
    assert len(rows)==len(old)==plan['horizon']
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(status='complete',experiment=plan['id'],method=method,seed=seed,
        horizon=plan['horizon'],completed_rounds=len(rows),config=cfg,trajectory=rows,timing=timing,
        peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),
        parameters=dict(initial=initial,final=parameters(model)),transfer=getattr(model,'transfer',None),
        spectra=getattr(model,'spectral_history',[]),numerical_fallbacks=getattr(model,'numerical_fallbacks',[]),
        reference=dict(file=record['result_file'],sha256=record['source_sha256'],
            matching_actions_and_regret=not(differences['action'] or differences['regret']),
            matching_ranks=not differences['rank'],
            matching_rewards=(not differences['reward']) if all('reward' in r for r in old) else None,
            first_different_round=min((t for values in differences.values() for t in values),default=None)),
        summary=dict(final_rank=model.m if adaptive else None,regret=math.fsum(r['regret'] for r in rows),
            regret_after_100=math.fsum(r['regret'] for r in rows[100:]),
            regret_after_300=math.fsum(r['regret'] for r in rows[300:]))),model


def jobs(plan):
    if plan.get('kind')=='spectral_diagnostic':
        return [('Trained features',plan['seeds'][0])]
    if plan['paper_key'].startswith('physical_fold300.'):
        return [(m,s) for i,s in enumerate(plan['paper_seed_order'])
            for m in (('BBP','Late') if i%2==0 else ('Late','BBP'))]
    return [(r['method'],r['seed']) for r in sorted(plan['runs'],
        key=lambda r:(plan['paper_seed_order'].index(r['seed']),r['method']))]


def result_path(directory, plan, method, seed):
    return directory/plan['id']/f"seed{seed}_{method.replace(' ','_')}"/'result.json'


def main():
    parser = argparse.ArgumentParser(description='Run paper experiments.')
    parser.add_argument('--experiment')
    parser.add_argument('--method')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--from-checkpoint',action='store_true',help='Repeat BBP spectral sampling without retraining.')
    args = parser.parse_args()
    plans = protocols()
    if args.list:
        from plot_paper import method_name
        for name,p in plans.items():
            methods = ['Trained features'] if p.get('kind')=='spectral_diagnostic' else p['methods']
            print(f"{name}: {p['title']}\n  Methods: {', '.join(method_name(m, plain=True) for m in methods)}\n  Seeds: {', '.join(map(str,p['seeds']))}")
        return
    if args.all:
        if args.experiment or args.method or args.seed is not None or args.from_checkpoint:
            parser.error('Use --all by itself, optionally with --output.')
        directory = (args.output or DATA/'reruns').resolve()
        hashes = source_hashes()
        for plan in plans.values():
            for method,seed in jobs(plan):
                path = result_path(directory,plan,method,seed)
                if not path.exists():
                    subprocess.run([sys.executable,'-B',str(Path(__file__)),'--experiment',plan['id'],
                        '--method',method,'--seed',str(seed),'--output',str(path.parent)],check=True)
                value = read(path)
                if (value.get('status')!='complete' or value.get('experiment')!=plan['id'] or
                    value.get('method')!=method or value.get('seed')!=seed or
                    value.get('source_sha256')!=hashes or value.get('from_checkpoint',False)):
                    raise ValueError(f'Existing run differs from this configuration: {path}')
                if any(v is False for k,v in value.get('reference',{}).items() if k.startswith('matching_')):
                    raise ValueError(f'Run differs from saved results: {path}')
        return
    if args.experiment not in plans or args.seed is None:
        parser.error('Use --experiment EXPxxx --method METHOD --seed SEED, or --list/--all.')
    plan = plans[args.experiment]
    method = args.method or ('Trained features' if plan.get('kind')=='spectral_diagnostic' else None)
    if (method,args.seed) not in jobs(plan):
        parser.error('Choose a method and seed listed for this experiment.')
    if args.from_checkpoint and plan.get('kind')!='spectral_diagnostic':
        parser.error('--from-checkpoint applies only to the BBP spectral experiment.')
    output = (args.output or result_path(DATA/'reruns',plan,method,args.seed).parent).resolve()
    if output.is_relative_to(DATA) and not output.is_relative_to(DATA/'reruns'):
        parser.error('New runs inside paper_experiments must be under reruns/.')
    output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    hashes = source_hashes()
    try:
        if plan.get('kind')=='spectral_diagnostic':
            import paper_bbp
            origin = DATA/plan['folder']/'results'
            if args.from_checkpoint:
                for name in ('checkpoint.pt','calibration.npz'):
                    shutil.copy2(origin/name,output/name)
            else:
                paper_bbp.train(output)
            paper_bbp.sample(output)
            import numpy as np
            with np.load(origin/'samples.npz',allow_pickle=False) as old, np.load(
                    output/'classical_transition/samples.npz',allow_pickle=False) as new:
                same = set(old.files)==set(new.files) and all(
                    np.allclose(old[k],new[k],rtol=1e-9,atol=1e-9) for k in old.files)
            result = dict(status='complete',experiment=plan['id'],method=method,seed=args.seed,
                from_checkpoint=args.from_checkpoint,results='classical_transition/results.json',
                reference=dict(matching_spectral_samples=bool(same)))
        else:
            record = next(r for r in plan['runs'] if r['method']==method and r['seed']==args.seed)
            if sha(DATA/record['result_file'])!=record['source_sha256']:
                raise ValueError('Saved reference changed')
            result,model = train(plan,record)
            torch.save(checkpoint(model),output/'checkpoint.pt')
        assert hashes==source_hashes(),'Numerical code changed during run'
        result['source_sha256'] = hashes
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output),status='complete',reference=result.get('reference'))))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
