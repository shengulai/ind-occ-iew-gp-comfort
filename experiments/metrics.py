import numpy as np
from .settings import PERSONAS

def scores(mean, variance, target):
    mean, target = np.asarray(mean, dtype=float), np.asarray(target, dtype=float)
    error = target - mean
    result = dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))))
    result['nlpd'] = None if variance is None else float(np.mean(
        .5 * np.log(2*np.pi*np.asarray(variance, dtype=float)) + .5*error**2/variance))
    return result

def write_table(path, rows):
    labels = {'default_iew':'Default IEW','calibrated_iew':'Calibrated IEW',
              'constant_gp':'Constant-mean GP','initial_gp':'Initialized IEW + GP','proposed_gp':'Learned IEW + GP'}
    lookup = {(r['persona'],r['model'],r['reference']):r for r in rows}
    lines = [r'\begin{table}[htbp]',r'\centering\small',
             r'\caption{Point and latent probabilistic performance against the predefined IEW reference and the reference with a shared sampled residual. NLPD excludes test observation noise and is not reported for deterministic models.}',
             r'\label{tab:comfort_model_metrics}',r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{0.85\columnwidth}{!}{\begin{tabular}{lrrrrrr}',r'\toprule',
             r' & \multicolumn{3}{c}{IEW reference} & \multicolumn{3}{c}{IEW reference with residual} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Model & MAE & RMSE & NLPD & MAE & RMSE & NLPD \\']
    for persona in PERSONAS:
        lines.append(r'\midrule')
        for model,label in labels.items():
            values=[]
            for target in ('iew','iew_plus_residual'):
                values.extend('---' if lookup[persona,model,target][key] is None else f'{lookup[persona,model,target][key]:.4f}' for key in ('mae','rmse','nlpd'))
            name=f'{persona} ({label})'
            if model=='proposed_gp':name=r'\textbf{'+name+'}'
            lines.append(name+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')
