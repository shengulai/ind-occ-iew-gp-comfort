"""CSV example: python -m comfort_model --help."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from . import ComfortModel, interpret_feedback


def main():
    p = argparse.ArgumentParser(description='Fit comfort from saved scores or optionally interpret natural-language feedback.')
    p.add_argument('--feedback', type=Path, required=True, help='CSV: timestamp, temperature_F, co2_ppm, comfort (or feedback with --interpret).')
    p.add_argument('--conditions', type=Path, required=True, help='CSV: temperature_F, co2_ppm to predict.')
    p.add_argument('--output', type=Path, default=Path('outputs/example'))
    p.add_argument('--interpret', action='store_true', help='Call OpenAI on each feedback comment instead of using comfort column.')
    p.add_argument('--examples', type=Path, help='Optional plain-text language examples for the interpreter.')
    p.add_argument('--llm-model', default='gpt-4o')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--tau-days', type=float, default=120.)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose a new/empty output directory.')
    with a.feedback.open() as f:
        rows = list(csv.DictReader(f))
    with a.conditions.open() as f:
        conditions = list(csv.DictReader(f))
    if not rows or not conditions:
        p.error('Feedback and prediction CSVs must be nonempty.')
    x = [[float(r['temperature_F']), float(r['co2_ppm'])] for r in rows]
    dates = [datetime.fromisoformat(r['timestamp']) for r in rows]
    ages = [(max(dates)-d).total_seconds()/86400 for d in dates]
    query = [[float(r['temperature_F']), float(r['co2_ppm'])] for r in conditions]
    # Validate numerical inputs before making any optional paid API calls.
    ComfortModel._inputs(x)
    ComfortModel._inputs(query)
    model = ComfortModel(steps=a.steps, tau_days=a.tau_days)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.interpret:
        examples = a.examples.read_text() if a.examples else ''
        from openai import OpenAI
        client = OpenAI()
        y = []
        # Persist each successful result so it remains usable if a later call fails.
        with (a.output/'interpreted_feedback.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['timestamp', 'temperature_F', 'co2_ppm', 'feedback',
                                              'severity', 'comfort', 'response', 'model'])
            w.writeheader()
            for r in rows:
                result = interpret_feedback(r['feedback'], examples=examples, model=a.llm_model, client=client)
                y.append(result.comfort)
                w.writerow({**{k:r[k] for k in ('timestamp','temperature_F','co2_ppm','feedback')},
                            'severity':result.severity, 'comfort':result.comfort,
                            'response':result.response, 'model':result.model})
                f.flush()
    else:
        y = [float(r['comfort']) for r in rows]
    model.fit(x, y, ages_days=ages)
    model.save(a.output/'comfort_model.pt')
    pred = model.predict(query)
    with (a.output/'predictions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['temperature_F','co2_ppm',*pred]);w.writeheader()
        for i,r in enumerate(conditions):
            w.writerow({**{k:r[k] for k in ('temperature_F','co2_ppm')}, **{k:float(v[i]) for k,v in pred.items()}})
    print(f'Saved model and predictions in {a.output}')


if __name__ == '__main__':
    main()
