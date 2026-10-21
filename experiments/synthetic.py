"""Synthetic persona generator; original sampling, reporting rule and language prompt."""
import numpy as np
import torch
import gpytorch
from comfort_model.gp import ComfortBaseline

class SyntheticPersona:
    def __init__(self, name, TRUE_PARAMS, character_info, llm_model='gpt-4o', client=None):
        self.name = name
        self.TRUE_PARAMS = TRUE_PARAMS
        self.character_info = character_info
        self.llm_model = llm_model
        self.client = client

    def determine_impact_factor(self, cur_temp, cur_co2):
        output = []

        if cur_temp > self.TRUE_PARAMS['wellness_range']['temp'][1]:
            output.append("feels warmer than comfort")
        elif cur_temp < self.TRUE_PARAMS['wellness_range']['temp'][0]:
            output.append("feels cooler than comfort")

        if cur_co2 > self.TRUE_PARAMS['wellness_range']['co2'][1]:
            output.append("co2 higher than comfort zone, feels stuffy/dizzey, want fresh air")

        if cur_co2 < 500:
            output.append("You are not likely to complain about co2")
        if cur_temp < self.TRUE_PARAMS['wellness_range']['temp'][1] - 2:
            output.append("You are not likely to complain about temp being warm")
        if cur_temp > self.TRUE_PARAMS['wellness_range']['temp'][0] + 2:
            output.append("You are not likely to complain about temp being cold")

        return output

    def score_to_sensation_level(self, comfort_score):
        threshold = self.TRUE_PARAMS['feedback_params']['complaint_threshold']
        if comfort_score >= 0.95:
            return "perfect condition, very comfortable"
        elif comfort_score > threshold:
            return "ok, bearable, maybe be slightly uncomfortable"
        elif comfort_score > min(threshold - 0.1, 0.6):
            return "moderate discomfort"
        elif comfort_score > min(threshold - 0.2, 0.5):
            return "noticeable discomfort"
        else:
            return "very uncomfortable"

    async def generate_human_response(self, temp, co2, comfort_score):
        factor_impact = self.determine_impact_factor(temp, co2)

        comfort_level = self.score_to_sensation_level(comfort_score)

        prompt = f"""
        Context information:
        You are a building occupant, using the a lab/office space at Harvard University, Harvard Center for Green Buildings and Cities. You are a user of the Occupied App, which allows you to provide natural language feedback on the your comfort sensations.
        
        Your character information: {self.character_info}

        Current conditions:
        - Your current comfort level: {comfort_level}
        - If there is discomfort, it is likely to be caused by: {factor_impact}
        
        Your task:
        Based on your character and your tone, Generate ONE realistic feedback statement that this person might give.
        
        The feedback should:
        1. Align with the character's information provided
        2. Reflect the comfort level indicated, but not knowing the details of what the comfort model is looking for
        3. If condition is not perfect, mention specificly what bothers you (e.g.: temperature -> hot/cold, co2 -> fresh air/dizzy) only if you know
            3.1 If comfort level is perfect, ignore the assumed cause above
            3.2 If comfort level is not perfect, but there is no determined cause, don't mention specific factors, just indicate your discomfort
        4. Be varied and improvise with different phrases
        5. These feedback are typically concise 1 liners, rarely has multiple sentences
        """

        if self.client is None:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI()
        response = await self.client.chat.completions.create(
            model=self.llm_model, temperature=.7, messages=[{'role':'user','content':prompt}])
        text = response.choices[0].message.content
        if not text:
            raise ValueError('No generated feedback was returned.')
        return text

    def generate_ground_truth_model(self):
        """
        Build the ground-truth ComfortBaseline + Likelihood with frozen, true parameters.
        """
        baseline = ComfortBaseline(
            weights=self.TRUE_PARAMS['weights'],
            wellness_range=self.TRUE_PARAMS['wellness_range'],
        )
        for param in baseline.parameters():
            param.requires_grad = False

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        with torch.no_grad():
            likelihood.noise = self.TRUE_PARAMS['gp_params']['noise_var']
        likelihood.noise_covar.raw_noise.requires_grad = False

        return baseline, likelihood

    def sample_residual(self, X_sample,
                        lengthscale_temp=None,
                        lengthscale_co2=None,
                        signal_var=None):
        """Sample a GP residual at X_sample (rows = [temp, co2])."""
        if lengthscale_temp is None:
            lengthscale_temp = self.TRUE_PARAMS['gp_params']['lengthscale_temp']
        if lengthscale_co2 is None:
            lengthscale_co2 = self.TRUE_PARAMS['gp_params']['lengthscale_co2']
        if signal_var is None:
            signal_var = self.TRUE_PARAMS['gp_params']['signal_var']

        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        kernel.outputscale = signal_var
        kernel.base_kernel.lengthscale = torch.tensor([lengthscale_temp, lengthscale_co2])

        with torch.no_grad():
            covar = kernel(X_sample)
            dist = gpytorch.distributions.MultivariateNormal(
                torch.zeros(X_sample.shape[0]),
                covar,
            )
            gp_residual = dist.sample()

        return gp_residual

    def sample_comfort_total(self, X_sample, debug=False):
        """Ground-truth comfort = IEW baseline + GP residual + observation noise (clamped 0..1)."""
        baseline_true, _ = self.generate_ground_truth_model()
        iew_score = baseline_true(X_sample, occ_rate=1.0)
        gp_residual = self.sample_residual(
            X_sample,
            lengthscale_temp=self.TRUE_PARAMS['gp_params']['lengthscale_temp'],
            lengthscale_co2=self.TRUE_PARAMS['gp_params']['lengthscale_co2'],
            signal_var=self.TRUE_PARAMS['gp_params']['signal_var'],
        )
        noise = torch.randn(X_sample.shape[0]) * torch.sqrt(
            torch.tensor(self.TRUE_PARAMS['gp_params']['noise_var'])
        )

        comfort_total = torch.clamp(iew_score + gp_residual + noise, 0.0, 1.0)

        if debug:
            print("iew_score: ", iew_score)
            print("gp_residual: ", gp_residual)
            print("noise: ", noise)
            print("comfort_total: ", comfort_total)

        return comfort_total

    def calculate_feedback_probability(self, comfort_score,
                                       complaint_threshold=None,
                                       p_at_threshold=None,
                                       p_max=None,
                                       p_min=None,
                                       decay_rate=None):
        """
        Linear below threshold, exponential decay above threshold (clipped to [p_min, p_max]).
        """
        fp = self.TRUE_PARAMS['feedback_params']
        if complaint_threshold is None:
            complaint_threshold = fp['complaint_threshold']
        if p_at_threshold is None:
            p_at_threshold = fp['p_at_threshold']
        if p_max is None:
            p_max = fp['p_max']
        if p_min is None:
            p_min = fp['p_min']
        if decay_rate is None:
            decay_rate = fp['decay_rate']

        if comfort_score < complaint_threshold:
            prob = p_at_threshold + (p_max - p_at_threshold) * (
                complaint_threshold - comfort_score
            ) / complaint_threshold
        else:
            delta = comfort_score - complaint_threshold
            prob = p_at_threshold * np.exp(-decay_rate * delta)

        return np.clip(prob, p_min, p_max)
