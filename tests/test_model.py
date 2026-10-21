import tempfile
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from comfort_model import ComfortModel, interpret_feedback
from comfort_model.language import parse_severity


class ModelTest(unittest.TestCase):
    def test_fit_update_and_checkpoint_round_trip(self):
        torch.set_num_threads(1)
        m=ComfortModel(steps=20).fit([[68,500],[73,800],[79,1200]],[.5,.95,.3],ages_days=[10,5,0])
        q=[[70,600],[75,900]]
        p=m.predict(q)
        self.assertTrue(np.all(np.isfinite(p['mean'])))
        self.assertTrue(np.all(p['variance']>0))
        self.assertTrue(np.all((p['comfort']>=0)&(p['comfort']<=1)))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';m.save(path)
            restored=ComfortModel.load(path)
            for k,v in p.items():np.testing.assert_allclose(restored.predict(q)[k],v,rtol=1e-6,atol=1e-7)
            restored.update(74,700,.9,elapsed_days=2)
            np.testing.assert_array_equal(restored.ages_days,[12,7,2,0])
            self.assertEqual(len(restored.model.train_targets),4)

    def test_reject_invalid_pairs(self):
        for x,y,ages in [([[72,800]],[1.1],[0]),([[72,-1]],[.5],[0]),
                         ([[72,800]],[.5],[-1]),([[float('nan'),800]],[.5],[0])]:
            with self.assertRaises(ValueError):ComfortModel(steps=1).fit(x,y,ages_days=ages)

    def test_interpretation_and_prompt_with_mock_client(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='A little uncomfortable. [0.2]'))])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        answer=interpret_feedback('It is a little warm.',examples='A little means mild discomfort.',client=client)
        self.assertEqual(answer.comfort,.8)
        self.assertEqual(answer.severity,.2)
        self.assertEqual(calls[0]['model'],'gpt-4o')
        self.assertEqual(calls[0]['temperature'],.7)
        self.assertIn('A little means mild discomfort.',calls[0]['messages'][0]['content'])
        for response in ('No score','[1.1]','[True]','[0.2, 0.3]','[0.2] then [0.3]'):
            with self.assertRaises(ValueError):parse_severity(response)

    def test_synthetic_generation_with_mock_client(self):
        from experiments.synthetic import SyntheticPersona
        from experiments.settings import GENERATION
        calls=[]
        async def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='It feels warm in here.'))])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        definition=GENERATION['Adam']
        npc=SyntheticPersona('Adam',definition['parameters'],definition['character_info'],client=client)
        torch.manual_seed(0)
        score=float(npc.sample_comfort_total(torch.tensor([[76.,800.]])))
        self.assertTrue(0<=score<=1)
        self.assertTrue(0<=npc.calculate_feedback_probability(score)<=1)
        self.assertEqual(asyncio.run(npc.generate_human_response(76,800,score)),'It feels warm in here.')
        self.assertEqual(calls[0]['model'],'gpt-4o')
        self.assertIn('Current conditions:',calls[0]['messages'][0]['content'])


if __name__=='__main__':unittest.main()
