import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'paper_experiments'
BASELINES = ('NN-AGP (m=7)', 'GP-UCB (product)', 'GP-UCB (weighted sum)', 'NeuralUCB', 'NN-UCB')


def read(path):
    return json.loads(Path(path).read_text())


def protocols():
    return {p.parent.name: read(p) for p in sorted(DATA.glob('*/EXP*/protocol.json'))}


def configuration(plan, record):
    cfg = copy.deepcopy(plan['configurations'][record['configuration']])
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
        env = paper_envs.WeakBalancedEnv(spec, seed, plan['horizon'])
    else:
        settings = copy.deepcopy(cfg)
        nuisance = plan['paper_key']=='online.smooth3_128'
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
        physical = plan['paper_key'].startswith('physical.')
        return NN_AGP_UCB_Adaptive.from_config(cfg, seed, 'BBP' if method=='BBP + physical compression' else method,
            calibration='fresh' if plan['environment'] in ('Weak','Balanced') else 'history',
            history_svd=plan['settings'].get('history_svd',False),
            corrected_svd=plan['settings'].get('corrected_svd_policy',True),
            fold_round=plan['settings']['fold_round'] if physical and method=='BBP + physical compression' else None,
            catalog=physical)
    d, x = cfg['env']['d_theta'], cfg['env']['d_x']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if method=='NN-AGP (m=7)':
        from bandits.nn_agp import NN_AGP_UCB
        return NN_AGP_UCB(d,x,m=math.ceil(d/3+x/10)+3,hidden_dim=64,lr=1e-3,
            mll_steps=40,beta=2.,init_noise=cfg['agent']['noise_sd'],dtype=torch.float64,batched=True)
    if method in ('GP-UCB (product)','GP-UCB (weighted sum)'):
        from bandits.gp_ucb_multikernel import GP_UCB_KernelFlexible
        return GP_UCB_KernelFlexible(x,d,cfg['agent']['noise_sd'],beta=2.,
            kernel_type='multiplicative' if method=='GP-UCB (product)' else 'adaptive',
            adaptive_weights={'theta':.6,'x':.4})
    if method=='NeuralUCB':
        from bandits.neural_ucb import NeuralUCB
        return NeuralUCB(input_dim=d+x,T=plan['horizon'],lambd=1.,nu=1.,delta=.01,
            S=1.,eta=1e-3,J=20,m=64,L=2,dual=True)
    from bandits.nn_ucb import NN_UCB
    return NN_UCB(context_dim=d+x,hidden_width=256,hidden_layers=2,lambda_=1.,
        beta=1.,J=10,lr=1e-3,seed=seed,device='cpu')


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
        if t==1 and method in ('GP-UCB (product)','GP-UCB (weighted sum)'):
            order = [m for m in np.roll(BASELINES,(1+seed)%len(BASELINES))
                if m in ('GP-UCB (product)','GP-UCB (weighted sum)')]
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
        rank = model.m if adaptive or method=='NN-AGP (m=7)' else None
        row = dict(t=t,action=action,theta=theta.tolist(),chosen_x=X[action].tolist(),
            reward=reward,regret=float(rewards.max()-rewards[action]),rank=rank,
            nn_width=getattr(model,'m_max',None),epsilon=getattr(model,'epsilon',None))
        if method=='NeuralUCB':
            model.update(None,np.hstack((theta,X[action])),reward)
        elif method=='NN-UCB':
            model.update(np.hstack((theta,X[action])),action,reward)
        elif method in BASELINES:
            model.update(theta,X[action],reward)
        else:
            calibration = None
            if fresh and t>=100 and t%cfg['agent']['T_spec']==0 and (
                    method!='BBP + physical compression' or t<=plan['settings']['fold_round']):
                calibration = env.calibration(seed,t,max(128,t))
            model.update(theta,X[action],reward,calibration_contexts=calibration)
        row['rank_after'] = model.m if adaptive else rank
        rows.append(row)
    timing = dict(cpu=time.process_time()-cpu,wall=time.perf_counter()-wall)
    old = read(DATA/record['result_file'])['trajectory']
    differences = {name:[r['t'] for r,s in zip(rows,old) if name in s and
        (r[name]!=s.get(name) if name in ('action','rank') else not math.isclose(r[name],s[name],rel_tol=1e-9,abs_tol=1e-9))]
        for name in ('action','rank','regret','reward')}
    assert len(rows)==len(old)==plan['horizon']
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(status='complete',experiment=plan['id'],method=method,seed=seed,
        horizon=plan['horizon'],trajectory=rows,timing=timing,
        peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),
        parameters=dict(initial=initial,final=parameters(model)),transfer=getattr(model,'transfer',None),
        spectra=getattr(model,'spectral_history',[]),numerical_fallbacks=getattr(model,'numerical_fallbacks',[]),
        reference=dict(file=record['result_file'],
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
    if plan['paper_key'].startswith('physical.'):
        return [(m,s) for i,s in enumerate(plan['seeds'])
            for m in (('BBP','BBP + physical compression') if i%2==0 else ('BBP + physical compression','BBP'))]
    return [(r['method'],r['seed']) for r in sorted(plan['runs'],
        key=lambda r:(plan['seeds'].index(r['seed']),r['method']))]


def result_path(directory, plan, method, seed):
    name = re.sub(r'[^A-Za-z0-9.-]+', '_', method).strip('_')
    return directory/plan['id']/f'seed{seed}_{name}'/'result.json'


def main():
    parser = argparse.ArgumentParser(description='Run paper experiments.')
    parser.add_argument('--experiment')
    parser.add_argument('--method',help='Method name from --list.')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--from-checkpoint',action='store_true',help='Repeat BBP spectral sampling without retraining.')
    args = parser.parse_args()
    plans = protocols()
    if args.list:
        for name,p in plans.items():
            methods = ['Trained features'] if p.get('kind')=='spectral_diagnostic' else p['methods']
            print(f"{name}: {p['environment']}\n  Methods: {', '.join(methods)}\n  Seeds: {', '.join(map(str,p['seeds']))}")
        return
    if args.all:
        if args.experiment or args.method or args.seed is not None or args.from_checkpoint:
            parser.error('Use --all by itself, optionally with --output.')
        directory = (args.output or DATA/'reruns').resolve()
        for plan in plans.values():
            for method,seed in jobs(plan):
                path = result_path(directory,plan,method,seed)
                if not path.exists():
                    subprocess.run([sys.executable,'-B',str(Path(__file__)),'--experiment',plan['id'],
                        '--method',method,'--seed',str(seed),'--output',str(path.parent)],check=True)
                value = read(path)
                if (value.get('status')!='complete' or value.get('experiment')!=plan['id'] or
                    value.get('method')!=method or value.get('seed')!=seed or
                    value.get('from_checkpoint',False)):
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
            result,model = train(plan,record)
            torch.save(checkpoint(model),output/'checkpoint.pt')
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output),status='complete',reference=result.get('reference'))))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
