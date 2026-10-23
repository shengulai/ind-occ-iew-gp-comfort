"""Recompute paper metrics or figures from saved data; no fitting or API calls."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comfort_model import ComfortModel
from comfort_model.gp import ComfortBaseline
from .settings import PERSONAS, REFERENCES, TEMPORAL, ADVERSARIAL
from .metrics import scores, write_table

DATA = Path(__file__).resolve().parents[1] / 'data'


def read_rows(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for k,v in row.items():
                if v == '': row[k] = None
                else:
                    try: row[k] = int(v)
                    except ValueError:
                        try: row[k] = float(v)
                        except ValueError: pass
            rows.append(row)
    return rows


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def verify(out):
    expected = {(r['persona'],r['model'],r['reference']):r for r in read_rows(DATA/'personas/table1_metrics.csv')}
    rows, paired_rows = [], []
    for persona in PERSONAS:
        e=arrays(DATA/f'personas/{persona}_evaluation.npz')
        pair=arrays(DATA/f'personas/{persona}_score_comparison.npz')
        history=arrays(DATA/f'personas/{persona}_learning.npz')
        np.testing.assert_array_equal(history['inputs'],pair['inputs'])
        np.testing.assert_array_equal(history['targets'],pair['llm_checkpoint_targets'])
        if persona=='Adam': assert history['targets'][39]==1.0
        for model in ('default_iew','calibrated_iew','constant_gp','initial_gp','proposed_gp'):
            for target in ('iew','iew_plus_residual'):
                result=scores(e[model+'_mean'], e.get(model+'_latent_variance'),e['reference_'+target])
                for key,value in result.items():
                    if value is not None:
                        assert f'{value:.4f}'==f'{expected[persona,model,target][key]:.4f}',(persona,model,target,key)
                rows.append(dict(persona=persona,model=model,reference=target,**result))
        for target in ('iew','iew_plus_residual'):
            np.testing.assert_array_equal(pair['reference_'+target],e['reference_'+target])
            for arm in ('llm_checkpoint','sampled_numerical'):
                paired_rows.append(dict(persona=persona,training_target=arm,reference=target,
                                        **scores(pair[arm+'_mean'],pair[arm+'_latent_variance'],pair['reference_'+target])))
        np.testing.assert_array_equal(pair['llm_checkpoint_mean'],e['proposed_gp_mean'])
        model=ComfortModel.load(DATA/f'personas/{persona}_model.pt')
        prediction=model.predict(e['grid'])
        np.testing.assert_allclose(prediction['mean'],e['proposed_gp_mean'],rtol=1e-5,atol=2e-6)
        np.testing.assert_allclose(prediction['variance'],e['proposed_gp_latent_variance'],rtol=1e-5,atol=2e-6)
        print(f'{persona}: Table 1, saved predictions and training pairs match.')
    write_rows(out/'table1_metrics.csv',rows)
    write_rows(out/'score_comparison_metrics.csv',paired_rows)
    write_table(out/'table1.tex',rows)
    from . import temporal, adversarial
    a=SimpleNamespace(**TEMPORAL)
    for seed in a.seeds:
        for scenario in a.scenarios:
            _,x,latent,y,mask=temporal.simulation(seed,scenario,a)
            saved=arrays(DATA/f'temporal/inputs/seed_{seed}_{scenario}.npz')
            for k,v in [('X',x),('latent',latent),('observed',y),('reported',mask)]:
                np.testing.assert_array_equal(v,saved[k])
    _,runs,_,_=adversarial.prepare(SimpleNamespace(**ADVERSARIAL))
    for seed,run in runs.items():
        saved=arrays(DATA/f'adversarial/inputs/seed_{seed}.npz')
        for key in ('X','latent','unmodified','corrupted','mask','attack_days','target'):
            np.testing.assert_array_equal(run[key],saved[key])
    print('All 30 temporal and 10 severe-error simulation inputs reproduce exactly.')
    from .test_pmv import verify_saved
    verify_saved(out/'pmv')


def scatter_figures(out):
    labels=['Default IEW','Calibrated IEW','Constant-mean GP','Learned IEW + GP']
    models=['default_iew','calibrated_iew','constant_gp','proposed_gp']
    colors=['#777777','#D55E00','#009E73','#0072B2']
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
    for persona in PERSONAS:
        e=arrays(DATA/f'personas/{persona}_evaluation.npz')
        fig,axes=plt.subplots(2,4,figsize=(8,3.9),sharex=True,sharey=True)
        fig.subplots_adjust(left=.075,right=.99,bottom=.13,top=.86,wspace=.10,hspace=.50)
        for j,target in enumerate(('iew','iew_plus_residual')):
            for i,(model,label,color) in enumerate(zip(models,labels,colors)):
                ax=axes[j,i];y=e[model+'_mean'];x=e['reference_'+target]
                result=scores(y,None,x)
                ax.scatter(x,y,s=3,alpha=.2,color=color,rasterized=True)
                ax.plot([0,1],[0,1],'--',color='#444444',lw=1)
                ax.text(.035,.97,f'MAE = {result["mae"]:.4f}\nRMSE = {result["rmse"]:.4f}',transform=ax.transAxes,va='top',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
                ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xticks=[0,.5,1],yticks=[0,.5,1])
                ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
                if j==0:ax.set_title(label,fontsize=10)
        for y,label in zip([.965,.475],['IEW reference','IEW + sampled residual']):
            fig.text(.075,y,label,fontsize=11,fontweight='bold')
        fig.supxlabel('Reference comfort',fontsize=11,y=.015);fig.supylabel('Predicted comfort',fontsize=11,x=.005)
        for ext in ('png','pdf'):fig.savefig(out/f'{persona}_final_comparison.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)


def figures(out):
    from . import evolution_plot, bounds_plot, uncertainty, temporal, adversarial
    scatter_figures(out)
    histories=[]; bounds=[]; calibration=[]
    temperature=np.linspace(60,85,250,dtype=np.float32)
    for persona,co2 in zip(PERSONAS,(1000,900,800)):
        history=dict(persona=persona,**arrays(DATA/f'personas/{persona}_learning.npz'))
        histories.append(history)
        evolution_plot.plot_persona(history,SimpleNamespace(snapshot_count=4,snapshot_height=1.25,dpi=300),out)
        bounds.append(dict(persona=persona,**arrays(DATA/f'personas/{persona}_score_bounds.npz')))
        e=arrays(DATA/f'personas/{persona}_evaluation.npz')
        model=ComfortModel.load(DATA/f'personas/{persona}_model.pt')
        x=np.column_stack([temperature,np.full_like(temperature,co2)])
        pred=model.predict(x)
        params=REFERENCES[persona]
        ref=ComfortBaseline(params['weights'],params['wellness_range'])
        with torch.no_grad():reference=ref(torch.tensor(x),params['occ_rate']).numpy()
        rows=[dict(persona=persona,target=target,**uncertainty.coverage_metrics(
            e['proposed_gp_mean'],e['proposed_gp_latent_variance'],e['reference_'+target],level))
              for target in ('iew','iew_plus_residual') for level in (.5,.65,.8,.9,.95,.99)]
        calibration.append(dict(persona=persona,co2=co2,temperature=temperature,slice_mean=pred['mean'],
                                slice_variance=pred['variance'],slice_reference=reference,rows=rows))
    evolution_plot.plot_comparison(histories,SimpleNamespace(dpi=300),out)
    bounds_plot.draw_figures(bounds,out)
    uncertainty.draw_figure(calibration,out,11,6)
    write_rows(out/'coverage.csv',[r for d in calibration for r in d['rows']])
    t=DATA/'temporal'
    temporal.plots(read_rows(t/'metrics_by_checkpoint.csv'),read_rows(t/'metrics_by_run.csv'),
                   read_rows(t/'paired_comparisons.csv'),SimpleNamespace(**TEMPORAL),out)
    t=DATA/'adversarial'
    adversarial.plots(read_rows(t/'metrics_by_checkpoint.csv'),read_rows(t/'paired_differences.csv'),
                      read_rows(t/'metrics_by_run.csv'),SimpleNamespace(**ADVERSARIAL),out)
    print(f'Figures generated in {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['verify','figures'])
    p.add_argument('--output',type=Path,default=Path('outputs/paper'))
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Choose a new/empty output directory.')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    (verify if a.task=='verify' else figures)(a.output)


if __name__=='__main__':main()
