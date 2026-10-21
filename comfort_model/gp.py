"""IEW mean and exact residual GP used in the paper."""
import torch
import gpytorch
import torch.nn as nn

class ComfortBaseline(nn.Module):
    def __init__(self, weights, wellness_range, 
                learn_weights=True, 
                fixed_lower_bounds={'co2': 0.0}, 
                normalize_inputs=True):
        super().__init__()

        self.weights = weights
        self.wellness_range = wellness_range
        self.learn_weights = learn_weights
        self.fixed_lower_bounds = fixed_lower_bounds or {}  # Default: no fixed bounds
        self.normalize_inputs = normalize_inputs
        
        # Set factors and factor indices
        self.factors = list(self.weights.keys())
        self.factor_to_idx = {factor: idx for idx, factor in enumerate(self.factors)}

        # Store normalization ranges (danger bounds)
        if normalize_inputs:
            for factor in self.factors:
                bad_low = self.wellness_range[factor][2]
                bad_up = self.wellness_range[factor][3]
                self.register_buffer(f'{factor}_range_min', 
                                    torch.tensor([bad_low], dtype=torch.float32))
                self.register_buffer(f'{factor}_range_max', 
                                    torch.tensor([bad_up], dtype=torch.float32))

        # LEARNABLE or FIXED parameters
        for factor in list(self.weights.keys()):
            # Weight parameters (as before)
            if learn_weights:
                raw_weight = torch.logit(torch.tensor([self.weights[factor]], dtype=torch.float32))
                setattr(self, f'_raw_weight_{factor}', nn.Parameter(raw_weight))
            else:
                weight_value = torch.tensor([self.weights[factor]], dtype=torch.float32)
                self.register_buffer(f'_weight_{factor}', weight_value)

            # Comfort range parameters
            lower_init = self.wellness_range[factor][0]
            upper_init = self.wellness_range[factor][1]

            # Normalize initial bounds if needed
            if normalize_inputs:
                bad_low = self.wellness_range[factor][2]
                bad_up = self.wellness_range[factor][3]
                range_span = bad_up - bad_low
                lower_init = (lower_init - bad_low) / range_span
                upper_init = (upper_init - bad_low) / range_span
            
            if factor in self.fixed_lower_bounds:
                # For factors with fixed lower bound (e.g., CO2 at 0)
                # Only learn the upper bound, lower is fixed
                fixed_lower = self.fixed_lower_bounds[factor]

                # Normalize fixed lower if needed
                if normalize_inputs:
                    bad_low = self.wellness_range[factor][2]
                    bad_up = self.wellness_range[factor][3]
                    fixed_lower = (fixed_lower - bad_low) / (bad_up - bad_low)

                self.register_buffer(f'{factor}_lower_fixed', 
                                    torch.tensor([fixed_lower], dtype=torch.float32))
                
                # Learn only the upper bound
                setattr(self, f'{factor}_upper', 
                       nn.Parameter(torch.tensor([upper_init], dtype=torch.float32)))
                
            else:
                # Standard center + log_width parameterization
                center_init = (lower_init + upper_init) / 2.0
                width_init = upper_init - lower_init
                
                if width_init <= 0:
                    raise ValueError(f"Initial width for {factor} must be positive, got {width_init}")
                
                setattr(self, f'{factor}_center', 
                       nn.Parameter(torch.tensor([center_init], dtype=torch.float32)))
                setattr(self, f'_{factor}_log_width', 
                       nn.Parameter(torch.log(torch.tensor([width_init], dtype=torch.float32))))
        
            # FIXED parameters: danger bounds (buffers)
            self.register_buffer(f'{factor}_bad_low', 
                               torch.tensor([self.wellness_range[factor][2]], dtype=torch.float32))
            self.register_buffer(f'{factor}_bad_up', 
                               torch.tensor([self.wellness_range[factor][3]], dtype=torch.float32))

    def normalize(self, X):
        """Normalize inputs to [0, 1] based on danger bounds"""
        if not self.normalize_inputs:
            return X
        
        X_norm = X.clone()
        for i, factor in enumerate(self.factors):
            range_min = getattr(self, f'{factor}_range_min')
            range_max = getattr(self, f'{factor}_range_max')
            X_norm[:, i] = (X[:, i] - range_min) / (range_max - range_min)
        
        return X_norm

    def denormalize_value(self, factor, value_norm):
        """Convert normalized value back to original scale"""
        if not self.normalize_inputs:
            return value_norm
        
        range_min = getattr(self, f'{factor}_range_min')
        range_max = getattr(self, f'{factor}_range_max')
        
        return range_min + value_norm * (range_max - range_min)

    def get_weight(self, factor):
        """Get the constrained weight [0, 1] for a factor"""
        if self.learn_weights:
            raw_weight = getattr(self, f'_raw_weight_{factor}')
            return torch.sigmoid(raw_weight)
        else:
            # Return fixed weight directly
            return getattr(self, f'_weight_{factor}')
    
    def get_comfort_bounds(self, factor, denormalize=True):
        """
        Get lower and upper comfort bounds from center and width.
        Ensure lower < upper always holds.
        """
        if factor in self.fixed_lower_bounds:
            lower = getattr(self, f'{factor}_lower_fixed')
            upper = getattr(self, f'{factor}_upper')
        else:
            center = getattr(self, f'{factor}_center')
            log_width = getattr(self, f'_{factor}_log_width')
            width = torch.exp(log_width)
            lower = center - width / 2
            upper = center + width / 2
        
        # Denormalize if requested
        if denormalize and self.normalize_inputs:
            lower = self.denormalize_value(factor, lower)
            upper = self.denormalize_value(factor, upper)
        
        return lower, upper

    def get_bounds(self, factor):
        """Get the lower and upper bounds for a factor"""
        lower, upper = self.get_comfort_bounds(factor)
        bad_low = getattr(self, f'{factor}_bad_low')
        bad_up = getattr(self, f'{factor}_bad_up')
        return lower, upper, bad_low, bad_up

    def calculate_iew_pytorch(self, X, occ_rate):
        """
        PyTorch implementation of iewIndex logic for multiple factors
        Following the exact calc_iew.py methodology with dynamic softmax weights
        
        Args:
            X: [N, d] tensor where columns correspond to factors in self.factors order
            occ_rate: float or [N] tensor of occupancy rates
        
        Returns:
            iew_score: [N] tensor of combined comfort scores [0, 1]
        """
        # Apply occupancy weight
        if isinstance(occ_rate, torch.Tensor):
            weight_occ = 1 - torch.exp(-10.0 * occ_rate)
        else:
            weight_occ = 1 - torch.exp(torch.tensor(-10.0 * occ_rate))
        
        # Collect weighted sig scores for all factors
        weighted_sig_scores_list = []
        
        for factor in self.factors:
            idx = self.factor_to_idx[factor]
            factor_val = X[..., idx]  # Extract column for this factor
            # weight = getattr(self, f'weight_{factor}')
            weight = self.get_weight(factor)
            
            # Calculate factor score and get the weighted discomfort
            _, weighted_sig_score = self.calculate_factor_score(factor_val, factor, occ_rate)
            weighted_sig_scores_list.append(weighted_sig_score)
        
        # Stack weighted sig scores: shape [N, num_factors]
        weighted_sig_scores = torch.stack(weighted_sig_scores_list, dim=-1)
        
        # Calculate individual IWQ indices (comfort per factor)
        IWQ_individual = 1 - weighted_sig_scores  # [N, num_factors]
        
        # Calculate dynamic weights using softmax
        # weights_dynamic = exp(weighted_sig_scores) / sum(exp(weighted_sig_scores))
        weights_dynamic = torch.softmax(weighted_sig_scores, dim=-1)  # [N, num_factors]
        
        # Calculate total IWQ with dynamic weights
        # IWQ_total = sum(IWQ_individual * weights_dynamic)
        IWQ_total = torch.sum(IWQ_individual * weights_dynamic, dim=-1)  # [N]
        
        # Clamp to [0, 1] range
        iew_score = torch.clamp(IWQ_total, 0.0, 1.0)
        
        return iew_score, weights_dynamic

    def calculate_factor_score(self, factor_val, factor_name, occ_rate):
        """
        Calculate comfort score for a single factor
        
        Args:
            factor_val: [N] tensor of factor values
            factor_name: string name of the factor ('temp', 'co2', etc.)
            occ_rate: float or [N] tensor of occupancy rates
        
        Returns:
            factor_score: [N] tensor of comfort scores [0, 1] for this factor
            weighted_sig_score: [N] tensor of weighted discomfort scores
        """
        # Get parameters for this factor
        lower_bound, upper_bound = self.get_comfort_bounds(factor_name)
        bad_low = getattr(self, f'{factor_name}_bad_low')
        bad_up = getattr(self, f'{factor_name}_bad_up')
        weight = self.get_weight(factor_name)
        
        # Same iewIndex logic implemented in PyTorch
        # Case 1: value below comfort range
        err_below = torch.relu(lower_bound - factor_val)
        err_ratio_below = (err_below / (lower_bound - bad_low + 1e-6)) ** 2
        
        # Case 2: value above comfort range
        err_above = torch.relu(factor_val - upper_bound)
        err_ratio_above = (err_above / (bad_up - upper_bound + 1e-6)) ** 2
        
        # Total error ratio
        err_ratio_2 = err_ratio_below + err_ratio_above
        
        # Sigma score (discomfort score)
        sig_score = (torch.exp(err_ratio_2)**10 - 1) / (1 + torch.exp(err_ratio_2)**10)
        
        # Apply occupancy weight
        if isinstance(occ_rate, torch.Tensor):
            weight_occ = 1 - torch.exp(-10.0 * occ_rate)
        else:
            weight_occ = 1 - torch.exp(torch.tensor(-10.0 * occ_rate))
        
        weighted_sig_score = weight_occ * weight * sig_score
        
        # Check danger flag (outside bad bounds)
        is_danger = ((factor_val < bad_low) | (factor_val > bad_up)).float()
        weighted_sig_score = torch.where(is_danger > 0.5, 
                                        torch.ones_like(weighted_sig_score), 
                                        weighted_sig_score)
        
        # Factor comfort score: 1 - discomfort
        factor_score = 1 - weighted_sig_score
        
        return factor_score, weighted_sig_score

    def forward(self, X, occ_rate=1.0):
        """
        Forward pass: compute IEW-based comfort score for multiple factors
        
        Args:
            X: [N, d] tensor where columns correspond to factors in self.factors order
               e.g., if self.factors = ['temp', 'co2'], then X[:, 0] = temp, X[:, 1] = co2
            occ_rate: float or [N] tensor of occupancy rates
        
        Returns:
            comfort_scores: [N] tensor of IEW indices [0, 1]
        """
        comfort_scores, weights_dynamic = self.calculate_iew_pytorch(X, occ_rate)
        return comfort_scores


class PhysicsMean(gpytorch.means.Mean):
    def __init__(self, baseline: ComfortBaseline, occ_rate=1.0):
        super().__init__()
        self.baseline = baseline
        self.occ_rate = occ_rate

    def forward(self, X):
        return self.baseline(X, self.occ_rate)


class ResidualGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, baseline: ComfortBaseline, 
                occ_rate=1.0, signal_var_low=0.001, signal_var_high=0.1,
                lengthscale_low=0.02, lengthscale_high=0.2, lengthscale_init=0.1):
        """
        Initialize the ResidualGP model with the given parameters
        
        Args:
            X_train: [N, d] tensor of training inputs
            y_train: [N] tensor of training targets
            likelihood: gpytorch.likelihoods.GaussianLikelihood
            baseline: ComfortBaseline
            occ_rate: float or [N] tensor of occupancy rates
            signal_var_low: float, lower bound for signal variance
            signal_var_high: float, upper bound for signal variance
            lengthscale_low: float, lower bound for lengthscale, default is 2% of range
            lengthscale_high: float, upper bound for lengthscale, default is 20% of range
            lengthscale_init: float, initial lengthscale, default is 10% of range
        """

        super().__init__(X_train, y_train, likelihood)

        self.mean_module = PhysicsMean(baseline, occ_rate)
        self.baseline = baseline
        
        # Build product kernel: k_total = k_temp × k_co2 × k_rh × ...
        # Each factor gets its own RBF kernel with lengthscale based on its range
        kernels = []
        for factor in baseline.factors:
            # Determine the effective range for lengthscale constraints
            if baseline.normalize_inputs:
                # Normalized inputs: range is [0, 1]
                effective_range = 1.0
            else:
                # Original scale inputs
                bad_low = getattr(baseline, f'{factor}_bad_low').item()
                bad_up = getattr(baseline, f'{factor}_bad_up').item()
                effective_range = bad_up - bad_low
            
            # Create RBF kernel for this factor
            kernel = gpytorch.kernels.RBFKernel(
                active_dims=[baseline.factor_to_idx[factor]],
                lengthscale_constraint=gpytorch.constraints.Interval(
                    effective_range * lengthscale_low,
                    effective_range * lengthscale_high
                )
            )
            # Initialize lengthscale
            kernel.lengthscale = effective_range * lengthscale_init
            kernels.append(kernel)
        
        # Combine kernels as product
        if len(kernels) == 1:
            base_kernel = kernels[0]
        else:
            base_kernel = gpytorch.kernels.ProductKernel(*kernels)
        
        # Scale the combined kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=gpytorch.constraints.Interval(signal_var_low, signal_var_high), # signal variance hard constraints
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 40.0) # Gamma prior parameters
        )

        # for 0.001 to 0.1, max std ~32%; if set to 0.15, max std ~39%, tune based on need
        # for a=2, b=40, mean=2/40=0.05 (middle of 0.1 upper bound), mode=(2-1)/40=0.025

    def forward(self, X):
        X_norm = self.baseline.normalize(X)
        mean_x = self.mean_module(X)    # baseline model mean
        covar_x = self.covar_module(X_norm)  # GP covariance
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_model(X_train, y_train, 
            weights_ini, 
            wellness_range_ini,
            learn_weights_ini=False,                # Default: weights are not learnable  
            fixed_lower_bounds_ini={'co2': 0.0},    # Default: fixed lower bounds for CO2
            normalize_inputs=True,                  # Default: normalize inputs to [0, 1]
            days_since_feedback=None,
            tau=120,
            occ_rate=1.0,
            num_steps=1000, 
            lr=0.01, 
            lr_baseline=None,                       # baseline (iewIndex) parameters, relatively higher
            lr_gp=None,                             # GP kernel parameters, relatively lower
            lr_likelihood=None,                     # likelihood parameters (noise variance), relatively lowest
            noise_llm_init=0.005,                   # inital variance for LLM uncertainty
            noise_llm_low=0.001,                    # lower bound for LLM uncertainty
            noise_llm_high=0.05,                    # upper bound for LLM uncertainty
            noise_hist_high=0.15,                   # upper bound for historical data uncertainty
            signal_init_var=0.005,                  # initial signal variance
            signal_var_low=0.001,                   # lower bound for signal variance
            signal_var_high=0.1,                    # upper bound for signal variance
            lengthscale_low=0.02,                   # lower bound for lengthscale
            lengthscale_high=0.2,                   # upper bound for lengthscale
            lengthscale_init=0.1):                  # initial lengthscale (10% of range)
    """
    Fit GP model with iewIndex-based baseline and optional temporal weighting with exponential forgetting implemented as extended noise variance on each observation
    """

    device = X_train.device

    ## Initialize iewIndex-based baseline with your parameters
    baseline = ComfortBaseline(
        weights=weights_ini,
        wellness_range=wellness_range_ini,
        learn_weights=learn_weights_ini,
        fixed_lower_bounds=fixed_lower_bounds_ini,
        normalize_inputs=normalize_inputs
    ).to(device)

    ## Compute temporal weights using exponential forgetting
    if days_since_feedback is not None:
        days_tensor = torch.tensor(days_since_feedback, dtype=torch.float32, device=device)
        
        # Exponential decay: w(t) = exp(-ln(2) * t / tau)
        # Half-life: at t=tau, weight = 0.5
        decay_rate = torch.log(torch.tensor(2.0)) / tau
        temporal_weights = torch.exp(-decay_rate * days_tensor)
        
        print(f"\nTemporal weighting (half-life τ={tau} days):")
        for i, (days, weight) in enumerate(zip(days_since_feedback, temporal_weights)):
            print(f"  Point {i+1}: {days:.0f} days ago → weight = {weight:.3f}")
        
        # Compute per-observation noise based on temporal weights
        # Old data gets higher noise (less trusted)
        noise_variances = noise_llm_init + (noise_hist_high * (1 - temporal_weights))
        # noise_variances = noise_variances.clamp(noise_llm_low, noise_hist_high)
        
        print(f"\nPer-observation noise variances:")
        for i, (days, noise) in enumerate(zip(days_since_feedback, noise_variances)):
            print(f"  Point {i+1}: {days:.0f} days ago → noise_var = {noise:.4f} (std = {torch.sqrt(noise):.3f})")
        
        # Use FixedNoiseGaussianLikelihood for heteroskedastic noise
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_variances,
            learn_additional_noise=True
        ).to(device)

        with torch.no_grad():
            likelihood.second_noise = torch.tensor(noise_llm_low, dtype=torch.float32, device=device)
        
        likelihood.second_noise_covar.register_constraint(
            "raw_noise", 
            gpytorch.constraints.Interval(noise_llm_low, noise_llm_high)
        )

    else:
        # Equal weights - use standard homoskedastic Gaussian likelihood
        temporal_weights = torch.ones(len(y_train), device=device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(noise_llm_low, noise_llm_high)
        )
        likelihood.noise = torch.tensor(noise_llm_init, dtype=torch.float32, device=device)

    ## Initialize GP model
    # model = ResidualGP(X_train, y_train, likelihood, baseline, occ_rate=occ_rate).to(device)
    model = ResidualGP(X_train, y_train, likelihood, baseline, 
                   occ_rate=occ_rate,
                   signal_var_low=signal_var_low,
                   signal_var_high=signal_var_high,
                   lengthscale_low=lengthscale_low,
                   lengthscale_high=lengthscale_high,
                   lengthscale_init=lengthscale_init).to(device)
    
    ## Set initial signal variance
    with torch.no_grad():
        model.covar_module.outputscale = torch.tensor(signal_init_var, device=device)

    ## Set to training mode
    model.train()
    likelihood.train()

    # all_params = list(model.parameters()) + list(likelihood.parameters())
    # optimizer = torch.optim.Adam(all_params, lr=lr)
    if lr_baseline is None:
        lr_baseline = lr
    if lr_gp is None:
        lr_gp = lr
    if lr_likelihood is None:
        lr_likelihood = lr

    # Separate parameter groups
    baseline_params = list(baseline.parameters())
    gp_params = list(model.covar_module.parameters())  # Kernel parameters only
    likelihood_params = list(likelihood.parameters())

    # Create optimizer with different learning rates for each group
    optimizer = torch.optim.Adam([
        {'params': baseline_params, 'lr': lr_baseline},
        {'params': gp_params, 'lr': lr_gp},
        {'params': likelihood_params, 'lr': lr_likelihood}
    ])

    # Marginal log-likelihood for optimization
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Optional: print progress
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}/{num_steps}, Loss: {loss.item():.4f}")

    return model, likelihood, baseline
