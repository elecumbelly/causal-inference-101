---
title: "Lesson 20: Bayesian Causal Inference"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 20: Bayesian Causal Inference

## Opening Story: Uncertainty in Treatment Effects

A clinical trial estimates that a new drug reduces mortality, but the estimate is imprecise and earlier studies provide relevant information. How should those sources of uncertainty be combined, and how should the result guide a decision?

Bayesian methods provide posterior distributions over causal effects *conditional on the model, prior and identification assumptions*. They can make uncertainty and prior information explicit, but they do not turn an unidentified causal effect into an identified one.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the Bayesian approach to causal inference
2. Implement Bayesian regression for treatment effects
3. Compute posterior distributions
4. Conduct Bayesian sensitivity analysis
5. Interpret Bayesian causal estimates

---

## 20.1 The Bayesian Framework

### Prior, Likelihood, Posterior

$$P(\theta | Data) \propto P(Data | \theta) \times P(\theta)$$

- **Prior**: Our beliefs before seeing data
- **Likelihood**: The probability of data given parameters
- **Posterior**: Our beliefs after seeing data

### Advantages for Causal Inference

1. **Uncertainty quantification**: Full posterior distribution over effects
2. **Prior information**: Can incorporate domain knowledge
3. **Hierarchical models**: Natural handling of multi-level data
4. **Model comparison**: Bayesian model selection

---

## 20.2 Bayesian Regression for Treatment Effects

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

np.random.seed(42)
n = 500

# Generate data
X = np.random.normal(0, 1, n)
T = np.random.binomial(1, 0.5, n)
Y = 2 + 1.5 * T + 0.8 * X + np.random.normal(0, 1, n)

# Bayesian model
with pm.Model() as causal_model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_treatment = pm.Normal('beta_treatment', mu=0, sigma=10)
    beta_covariate = pm.Normal('beta_covariate', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood
    mu = alpha + beta_treatment * T + beta_covariate * X
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

    # Sample
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# Summary
print(az.summary(trace, var_names=['beta_treatment']))

# Posterior distribution
az.plot_posterior(trace, var_names=['beta_treatment'])
```

---

## 20.3 Bayesian Sensitivity Analysis

```python
# Sensitivity to prior choice
with pm.Model() as sensitive_model:
    # Weakly informative prior
    beta_weak = pm.Normal('beta', mu=0, sigma=10)

    # Strongly informative prior
    beta_strong = pm.Normal('beta', mu=0, sigma=1)

    # Compare posteriors
    # ... (implementation depends on specific sensitivity questions)
```

---

## 20.4 Common Mistakes

1. **Ignoring prior sensitivity**: Always check how priors affect results
2. **Poor convergence**: Use diagnostics like R-hat and ESS
3. **Over-interpreting posteriors**: Remember this is conditional on the model
4. **Computational issues**: Use appropriate samplers and tuning

---

## 20.5 Knowledge Check

### Multiple Choice

1. **The Bayesian posterior:**
   A) Is the probability of the hypothesis
   B) Is the probability of the data
   C) Combines prior and likelihood
   D) Is always equal to the frequentist estimate

2. **Priors in Bayesian analysis:**
   A) Are always subjective
   B) Can be objective or subjective
   C) Don't matter
   D) Are always informative

3. **Bayesian credible intervals:**
   A) Have the same interpretation as frequentist CIs
   B) Give the probability the parameter is in the interval
   C) Are always wider
   D) Are always narrower

4. **MCMC sampling:**
   A) Is always exact
   B) Approximates the posterior
   C) Requires no tuning
   D) Is always fast

5. **Bayesian model comparison:**
   A) Uses p-values
   B) Uses Bayes factors or WAIC/LOO
   C) Is not possible
   D) Always favors complex models

### Short Answer

6. **Explain how Bayesian methods quantify uncertainty in treatment effects.**

7. **What role do priors play in Bayesian causal inference?**

8. **How do you assess convergence of MCMC chains?**

9. **What are the advantages of Bayesian methods for small samples?**

10. **Give an example where Bayesian methods would be preferred.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Causal inference course: uncertainty and estimation**](https://www.youtube.com/playlist?list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0)

Use the estimation and sensitivity lectures as a foundation before adding Bayesian priors and posterior uncertainty.

**Active-viewing prompt:** Check whether the prior is on an observable quantity, a causal effect, or an unidentified bias parameter.
```

---

## 20.6 Summary

1. **Bayesian methods** provide full posterior distributions
2. **Priors** incorporate domain knowledge
3. **MCMC** samples from the posterior
4. **Credible intervals** have intuitive interpretation
5. **Model comparison** uses WAIC/LOO or Bayes factors

---

## 20.7 Further Reading

- Gelman, A. et al. (2013). *Bayesian Data Analysis*. CRC Press.
- Hill, J. (2011). "Bayesian Nonparametric Modeling for Causal Inference." *Journal of Computational and Graphical Statistics*.


---

## Worked Examples

### Example 1: Bayesian ATE with Prior Information

Incorporating domain knowledge about plausible treatment effect sizes through informative priors. If prior studies suggest an effect between 2-5%, using a prior centered at 3.5% improves efficiency without introducing substantial bias.

### Example 2: Bayesian Propensity Scores

Using Bayesian logistic regression for propensity scores provides uncertainty quantification that frequentist methods lack. Posterior distributions over propensity scores propagate into the final treatment effect estimate.

### Example 3: Bayesian Hierarchical Models

Estimating treatment effects across multiple sites with a hierarchical model that partially pools estimates toward the grand mean, improving estimates for sites with small samples.

### Example 4: Bayesian Model Averaging

Averaging across multiple causal models weighted by posterior probability, accounting for model uncertainty rather than conditioning on a single model.

---

## Diagnostics: Bayesian Methods

### Prior Predictive Checks

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def prior_predictive_check(n_samples=10000):
    """Check if priors generate reasonable predictions."""

    # Prior on treatment effect
    ate_prior = stats.norm(loc=0, scale=1).rvs(n_samples)

    # Prior predictive distribution
    y_pred = np.random.normal(ate_prior, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(ate_prior, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', label='No effect')
    axes[0].set_xlabel('Treatment Effect')
    axes[0].set_title('Prior on ATE')
    axes[0].legend()

    axes[1].hist(y_pred, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Outcome')
    axes[1].set_title('Prior Predictive Distribution')

    plt.tight_layout()
    return fig
```

### Posterior Diagnostics

```python
def posterior_diagnostics(samples):
    """Check MCMC convergence and mixing."""

    n = len(samples)
    first = samples[:n//5]
    last = samples[4*n//5:]

    z = (np.mean(first) - np.mean(last)) / np.sqrt(np.var(first)/len(first) + np.var(last)/len(last))

    print(f"Geweke diagnostic: z = {z:.3f}")
    print(f"(Should be |z| < 1.96 for convergence)")

    autocorr = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]

    ess = n / (1 + 2 * np.sum(autocorr[1:]))

    print(f"Effective sample size: {ess:.0f}")
    print(f"(Should be > 400 for reliable inference)")
```

---

## Limitations

- Prior sensitivity: Results may depend heavily on prior choice
- Computational cost: MCMC can be slow for large datasets
- Convergence issues: MCMC may not converge or mix well
- Interpretability: Priors may be hard to justify to non-Bayesian audiences

---

## Exercises

1. **Prior specification**: Choose priors for a treatment effect analysis. How sensitive are results to prior choice?
2. **MCMC implementation**: Implement a simple Bayesian regression using MCMC. Check convergence.
3. **Bayesian vs frequentist**: Compare Bayesian and frequentist estimates. When do they differ most?
4. **Model comparison**: Use Bayes factors to compare causal models.

---

## Projects

### Project 1: Bayesian Sensitivity Analysis
Implement Bayesian sensitivity analysis for unmeasured confounding.

### Project 2: Hierarchical Bayesian Model
Build a hierarchical Bayesian model for multi-site treatment effects.


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


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/20-bayesian-causal-inference.ipynb)
- [Download the practice lab](../labs/lab20-bayesian-causal-inference-practice.ipynb)
- [Download the lab solution](../solutions/lab20-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
