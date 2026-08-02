

---

## Additional Advanced Content

### Prior Sensitivity Analysis

When using informative priors, it is essential to conduct sensitivity analysis to determine how much the results depend on the specific prior chosen. A robust finding should be relatively stable across reasonable prior specifications.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def prior_sensitivity_analysis(data, prior_means, prior_sds, n_samples=10000):
    """Analyze sensitivity to prior specification."""

    results = {}

    for mean, sd in zip(prior_means, prior_sds):
        # Simulate posterior samples
        prior = stats.norm(loc=mean, scale=sd)

        # Simplified posterior calculation (normal-normal conjugate)
        n = len(data)
        data_mean = np.mean(data)
        data_var = np.var(data) / n

        posterior_var = 1 / (1/sd**2 + 1/data_var)
        posterior_mean = posterior_var * (mean/sd**2 + data_mean/data_var)

        posterior = stats.norm(loc=posterior_mean, scale=np.sqrt(posterior_var))

        results[f"Prior({mean}, {sd})"] = {
            'mean': posterior_mean,
            'std': np.sqrt(posterior_var),
            'ci_95': posterior.interval(0.95)
        }

    # Plot sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, res in results.items():
        x = np.linspace(res['mean'] - 3*res['std'], res['mean'] + 3*res['std'], 100)
        y = stats.norm.pdf(x, res['mean'], res['std'])
        ax.plot(x, y, label=name, linewidth=2)

    ax.set_xlabel('Treatment Effect')
    ax.set_ylabel('Density')
    ax.set_title('Prior Sensitivity Analysis')
    ax.legend()
    plt.tight_layout()

    return results, fig
```

### Bayesian Model Comparison

```python
def bayes_factor(model1_log_lik, model2_log_lik, prior_odds=1.0):
    """Calculate Bayes factor comparing two models."""

    # Bayes factor is ratio of marginal likelihoods
    bf_12 = np.exp(model1_log_lik - model2_log_lik)

    # Posterior odds
    posterior_odds = prior_odds * bf_12

    print(f"Bayes Factor (Model 1 vs Model 2): {bf_12:.2f}")
    print(f"Interpretation:", end=" ")

    if bf_12 > 100:
        print("Decisive evidence for Model 1")
    elif bf_12 > 30:
        print("Very strong evidence for Model 1")
    elif bf_12 > 10:
        print("Strong evidence for Model 1")
    elif bf_12 > 3:
        print("Moderate evidence for Model 1")
    elif bf_12 > 1:
        print("Anecdotal evidence for Model 1")
    else:
        print("Evidence favors Model 2")

    return bf_12
```

### Hierarchical Bayesian Models for Multi-Site Studies

```python
def hierarchical_model_example(site_data):
    """Example of hierarchical Bayesian model for multi-site studies."""

    # Site-level parameters
    n_sites = len(site_data)

    # Hyperpriors
    mu_0 = 0  # Prior mean for site effects
    sigma_0 = 1  # Prior SD for site effects
    tau_0 = 0.5  # Prior SD for between-site variation

    # Site-specific effects (partially pooled)
    site_effects = []

    for site in site_data:
        n = len(site)
        site_mean = np.mean(site)

        # Shrinkage factor
        weight = tau_0**2 / (tau_0**2 + sigma_0**2/n)

        # Partially pooled estimate
        pooled_effect = weight * site_mean + (1 - weight) * mu_0

        site_effects.append(pooled_effect)

    print("Partially Pooled Site Effects:")
    for i, effect in enumerate(site_effects):
        print(f"  Site {i+1}: {effect:.3f}")

    return site_effects
```

### Computational Techniques

```python
def mcmc_diagnostics(samples):
    """Comprehensive MCMC diagnostics."""

    n_chains = samples.shape[0] if len(samples.shape) > 1 else 1
    n_samples = samples.shape[1] if len(samples.shape) > 1 else len(samples)

    print("MCMC Diagnostics:")
    print("=" * 50)

    if n_chains > 1:
        # R-hat statistic
        chain_means = np.mean(samples, axis=1)
        chain_vars = np.var(samples, axis=1)

        W = np.mean(chain_vars)  # Within-chain variance
        B = n_samples * np.var(chain_means)  # Between-chain variance

        r_hat = np.sqrt((W + B/n_samples) / W)
        print(f"R-hat: {r_hat:.3f}")
        print(f"(Should be < 1.1 for convergence)")

    # Effective sample size
    if n_chains == 1:
        flat_samples = samples
    else:
        flat_samples = samples.flatten()

    n = len(flat_samples)
    autocorr = np.correlate(flat_samples - np.mean(flat_samples),
                           flat_samples - np.mean(flat_samples), mode='full')
    autocorr = autocorr[n-1:] / autocorr[n-1]

    # Sum autocorrelations until they become negative
    sum_autocorr = 0
    for i in range(1, n):
        if autocorr[i] < 0:
            break
        sum_autocorr += autocorr[i]

    ess = n / (1 + 2 * sum_autocorr)
    print(f"Effective Sample Size: {ess:.0f}")
    print(f"(Should be > 400 for reliable inference)")

    # Geweke diagnostic
    first = flat_samples[:n//5]
    last = flat_samples[4*n//5:]
    z = (np.mean(first) - np.mean(last)) / np.sqrt(np.var(first)/len(first) + np.var(last)/len(last))
    print(f"Geweke z-statistic: {z:.3f}")
    print(f"(Should be |z| < 1.96 for convergence)")

    return {'ess': ess, 'r_hat': r_hat if n_chains > 1 else None, 'geweke_z': z}
```
