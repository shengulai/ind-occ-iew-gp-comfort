"""Optional fresh fitting of the paper experiments; saved paper results are read-only."""
import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import gpytorch

from comfort_model import ComfortModel
from .settings import PERSONAS, INITIAL, TEMPORAL, ADVERSARIAL
from .reproduce import DATA, arrays, read_rows, write_rows
from .metrics import scores
from . import baselines, temporal, adversarial


def persona_models(a, out):
    rows=[]
    for persona in PERSONAS:
        cfg=INITIAL[persona]
        pairs=read_rows(DATA/f'personas/{persona}_training.csv')
        x=np.array([[r['temperature_F'],r['co2_ppm']] for r in pairs],dtype=np.float32)
        ages=np.array([r['days_since_feedback'] for r in pairs])
        llm=np.array([r['llm_checkpoint_comfort'] for r in pairs],dtype=np.float32)
        if persona=='Adam':assert llm[39]==1.0
        e=arrays(DATA/f'personas/{persona}_evaluation.npz')
        settings=SimpleNamespace(tau=120.,steps=a.steps,learning_rate=.001,base_variance=.005,history_variance=.15)
        for name in ('calibrated_iew','constant_gp','proposed_gp','sampled_numerical_gp'):
            torch.manual_seed(0)
            y=(np.array([r['sampled_numerical_comfort'] for r in pairs],dtype=np.float32)
               if name=='sampled_numerical_gp' else llm)
            if name in ('proposed_gp','sampled_numerical_gp'):
                model=ComfortModel(temperature_bounds=cfg['wellness_range_initial']['temp'][:2],steps=a.steps)
                model.config=cfg.copy()
                model.fit(x,y,ages_days=ages)
                model.save(out/f'{persona}_{name}.pt')
                prediction=model.predict(e['grid'])
                mean,var=prediction['mean'],prediction['variance']
            elif name=='calibrated_iew':
                model,_=baselines.fit_iew(torch.tensor(x),torch.tensor(y),cfg,a.steps,.001)
                with torch.no_grad():mean=model(torch.tensor(e['grid']),cfg['occ_rate']).numpy()
                var=None
            else:
                model,likelihood=baselines.fit_constant_gp(torch.tensor(x),torch.tensor(y),ages,cfg,settings)
                model.eval();likelihood.eval()
                with torch.no_grad(),gpytorch.settings.fast_pred_var(False):
                    pred=model(torch.tensor(e['grid']));mean,var=pred.mean.numpy(),pred.variance.numpy()
            np.savez_compressed(out/f'{persona}_{name}_predictions.npz',grid=e['grid'],mean=mean,
                                **({'latent_variance':var} if var is not None else {}))
            for target in ('iew','iew_plus_residual'):
                rows.append(dict(persona=persona,model=name,reference=target,steps=a.steps,
                                 **scores(mean,var,e['reference_'+target])))
            print(f'{persona}: fitted {name}',flush=True)
    write_rows(out/'final_metrics.csv',rows)


def severe_error(a,out):
    grid,runs,changes,days=adversarial.prepare(a)
    write_rows(out/'corrupted_feedback.csv',changes)
    rows=[]
    for seed,run in runs.items():
        for name,tau in adversarial.variants(a):
            for day in days:
                clean=adversarial.evaluate_fit(seed,name,tau,day,'unmodified',run,grid,a)
                corrupted=(dict(clean,branch='corrupted') if day<run['attack_days'][0] else
                           adversarial.evaluate_fit(seed,name,tau,day,'corrupted',run,grid,a))
                rows.extend([clean,corrupted])
            write_rows(out/'metrics_by_checkpoint.csv',rows)
            print(f'seed={seed}, {name} complete',flush=True)
    paired=adversarial.paired_rows(rows)
    summaries=adversarial.summarize(paired,runs,a)
    write_rows(out/'paired_differences.csv',paired)
    write_rows(out/'metrics_by_run.csv',summaries)
    adversarial.plots(rows,paired,summaries,a,out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['personas','temporal','adversarial'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,help='Default: 2000 persona / 1000 temporal optimizer steps.')
    p.add_argument('--seeds',nargs='+',type=int,help='Temporal/adversarial only; default 0 through 9.')
    p.add_argument('--quick',action='store_true',help='Short execution check, not manuscript evidence.')
    cli=p.parse_args()
    if cli.steps is not None and cli.steps<1:p.error('--steps must be positive.')
    if cli.seeds and (cli.experiment=='personas' or min(cli.seeds)<0 or len(set(cli.seeds))!=len(cli.seeds)):
        p.error('--seeds requires unique nonnegative integers for temporal/adversarial experiments.')
    if cli.output.exists() and any(cli.output.iterdir()):p.error('Choose a new/empty output directory.')
    cli.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if cli.experiment=='personas':
        cli.steps=cli.steps or (20 if cli.quick else 2000)
        persona_models(cli,cli.output)
    else:
        settings=dict(TEMPORAL if cli.experiment=='temporal' else ADVERSARIAL)
        settings.update(steps=cli.steps or (20 if cli.quick else 1000),
                        seeds=cli.seeds or ([0] if cli.quick else list(range(10))),quick=cli.quick)
        if cli.quick:
            settings.update(taus=[100],eval_every=180)
            if cli.experiment=='temporal':settings['scenarios']=['stable']
            else:settings['event_eval_every']=60
        a=SimpleNamespace(**settings)
        (temporal.run if cli.experiment=='temporal' else severe_error)(a,cli.output)
    print('Quick execution check only.' if cli.quick else 'Completed fresh fits; archived results were not modified.')


if __name__=='__main__':main()
