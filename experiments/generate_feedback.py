"""Generate a NEW synthetic language dataset with GPT-4o (requires an API key)."""
import argparse
import asyncio
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .settings import GENERATION, PERSONAS
from .synthetic import SyntheticPersona


async def generate(a):
    data=pd.read_csv(a.environment)
    dates=pd.to_datetime(data['timestamp_US_Eastern'],utc=True).dt.tz_convert('US/Eastern')
    data=data.set_index(dates).loc[a.start:a.end]
    data=data[(data.index.hour>=8)&(data.index.hour<=18)]
    if data.empty:raise ValueError('No reporting hours in the selected date range.')
    missing=data[['temperature_F','co2_ppm']].isna().any(axis=1)
    if missing.any() and a.missing_policy=='error':
        raise ValueError('Missing reporting-hour conditions; specify --missing-policy skip for this new experiment.')
    data=data.loc[~missing]
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    definition=GENERATION[a.persona]
    npc=SyntheticPersona(a.persona,definition['parameters'],definition['character_info'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with a.output.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['timestamp','temperature_F','co2_ppm','feedback',
                                      'sampled_comfort','numerical_seed','model','temperature','missing_policy'])
        w.writeheader()
        for date,row in data.iterrows():
            x=torch.tensor([[row['temperature_F'],row['co2_ppm']]],dtype=torch.float32)
            score=float(npc.sample_comfort_total(x))
            if np.random.rand() < npc.calculate_feedback_probability(score):
                text=await npc.generate_human_response(float(x[0,0]),float(x[0,1]),score)
                w.writerow(dict(timestamp=date.isoformat(),temperature_F=float(x[0,0]),co2_ppm=float(x[0,1]),
                                feedback=text,sampled_comfort=score,numerical_seed=a.seed,model='gpt-4o',
                                temperature=.7,missing_policy=a.missing_policy))
                f.flush();count+=1
    print(f'Saved {count} new feedback records to {a.output}; skipped {int(missing.sum())} missing hours.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--persona',choices=PERSONAS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--environment',type=Path,default=Path(__file__).resolve().parents[1]/'data/environment/selected_hourly_environment.csv')
    p.add_argument('--start',default='2024-06-01');p.add_argument('--end',default='2025-03-07')
    p.add_argument('--missing-policy',choices=['error','skip'],default='error')
    p.add_argument('--output',type=Path,required=True,help='New CSV file; never overwrites an existing file.')
    a=p.parse_args()
    if a.seed<0:p.error('--seed must be nonnegative.')
    if a.output.exists():p.error('Choose a new output CSV.')
    asyncio.run(generate(a))


if __name__=='__main__':main()
