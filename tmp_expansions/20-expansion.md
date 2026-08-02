

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
