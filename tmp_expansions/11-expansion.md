

---

## Worked Examples

### Example 1: Economics — Class Size Threshold

In Israeli schools, classes with more than 40 students are split. This creates a discontinuity in class size at the threshold. Comparing students just above and below 40 estimates the effect of class size on achievement.

### Example 2: Health — Kidney Transplant Priority

Patients with GFR < 20 are prioritized for kidney transplants. Comparing patients just above and below 20 estimates the effect of transplant priority on survival.

### Example 3: Policy — Scholarship Eligibility

Students with SAT scores above 1200 receive a scholarship. Comparing students just above and below 1200 estimates the effect of financial support on college completion.

### Example 4: Technology — Feature Rollout

A platform releases a new feature to users with > 1000 followers. Comparing users just above and below 1000 estimates the feature's effect on engagement.

---

## Diagnostics: RDD Validity

### McCrary Density Test

```python
import numpy as np
import matplotlib.pyplot as plt

def mccrary_density_test(running_var, cutoff, bandwidth=None):
    """Test for manipulation of the running variable."""

    if bandwidth is None:
        bandwidth = np.std(running_var) * 0.2

    below = running_var[running_var < cutoff]
    above = running_var[running_var >= cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(running_var, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=cutoff, color='r', linestyle='--', label='Cutoff')
    axes[0].set_xlabel('Running Variable')
    axes[0].set_title('Distribution of Running Variable')
    axes[0].legend()

    # Density estimate
    from sklearn.neighbors import KernelDensity

    x_grid = np.linspace(cutoff - 3*bandwidth, cutoff + 3*bandwidth, 200).reshape(-1, 1)

    kde_below = KernelDensity(bandwidth=bandwidth).fit(below.reshape(-1, 1))
    kde_above = KernelDensity(bandwidth=bandwidth).fit(above.reshape(-1, 1))

    log_density_below = kde_below.score_samples(x_grid)
    log_density_above = kde_above.score_samples(x_grid)

    axes[1].plot(x_grid, np.exp(log_density_below), label='Below cutoff', linewidth=2)
    axes[1].plot(x_grid, np.exp(log_density_above), label='Above cutoff', linewidth=2)
    axes[1].axvline(x=cutoff, color='r', linestyle='--', label='Cutoff')
    axes[1].set_xlabel('Running Variable')
    axes[1].set_title('Density at Cutoff')
    axes[1].legend()

    plt.tight_layout()

    density_below = np.exp(kde_below.score_samples([[cutoff]]))[0]
    density_above = np.exp(kde_above.score_samples([[cutoff]]))[0]
    ratio = density_above / density_below

    print(f"Density ratio at cutoff: {ratio:.3f}")
    print("Values close to 1 suggest no manipulation")

    return fig, ratio
```

### Covariate Balance at the Cutoff

```python
def covariate_balance_rdd(df, running_var, cutoff, covariates, bandwidth=None):
    """Check covariate balance at the discontinuity."""

    if bandwidth is None:
        bandwidth = np.std(df[running_var]) * 0.2

    near_cutoff = df[(df[running_var] >= cutoff - bandwidth) &
                     (df[running_var] <= cutoff + bandwidth)]

    below = near_cutoff[near_cutoff[running_var] < cutoff]
    above = near_cutoff[near_cutoff[running_var] >= cutoff]

    print("Covariate Balance at Discontinuity:")
    for cov in covariates:
        diff = above[cov].mean() - below[cov].mean()
        pooled_std = np.sqrt((above[cov].var() + below[cov].var()) / 2)
        smd = abs(diff) / pooled_std if pooled_std > 0 else 0
        status = "OK" if smd < 0.1 else "WARN"
        print(f"  {cov}: SMD = {smd:.3f} [{status}]")
```

---

## Interpretation Workshop

### Reading RDD Studies

Key questions:
1. **What is the running variable and cutoff?** Is the threshold clearly defined?
2. **Is there manipulation?** (McCrary test, covariate balance)
3. **What bandwidth was used?** Narrower = more local, wider = more power
4. **What polynomial order?** Higher order can introduce bias
5. **Is the result robust** to bandwidth and polynomial choices?

### Local vs Global Estimates

RDD estimates the effect **at the cutoff** — a local average treatment effect. This may not generalize to units far from the threshold.

---

## Practical Application

### Standard RDD Estimation

```python
import numpy as np
import statsmodels.api as sm

def rdd_estimate(df, running_var, outcome, cutoff, bandwidth=None, polynomial=1):
    """Estimate RDD effect using local polynomial regression."""

    df = df.copy()
    df['centered'] = df[running_var] - cutoff

    if bandwidth is not None:
        df = df[df['centered'].abs() <= bandwidth]

    df['treated'] = (df['centered'] >= 0).astype(int)

    if polynomial == 1:
        X = df[['centered', 'treated']]
    elif polynomial == 2:
        df['centered_sq'] = df['centered'] ** 2
        X = df[['centered', 'centered_sq', 'treated']]
    else:
        for p in range(1, polynomial + 1):
            df[f'centered_{p}'] = df['centered'] ** p
        X = df[[f'centered_{p}' for p in range(1, polynomial + 1)] + ['treated']]

    X = sm.add_constant(X)
    model = sm.OLS(df[outcome], X).fit()

    effect = model.params['treated']
    se = model.bse['treated']

    print(f"RDD Estimate: {effect:.4f} (SE: {se:.4f})")

    return model
```

### Robustness Checks

1. **Vary bandwidth**: Use triangular kernel, vary bandwidth from 0.5h to 2h
2. **Vary polynomial order**: Compare linear, quadratic, cubic
3. **Placebo cutoffs**: Test at cutoffs where no effect is expected
4. **Covariate adjustment**: Add covariates to the regression

---

## Limitations

- **Local estimate**: Only valid at the cutoff
- **Sensitivity to bandwidth**: Results can vary with bandwidth choice
- **Manipulation**: If units can manipulate the running variable, the estimate is biased
- **Density continuity**: Requires no bunching at the cutoff

---

## Exercises

1. **Implement RDD**: Use the provided code to estimate an RDD effect. Vary the bandwidth and polynomial order. How stable are the results?

2. **McCrary test**: Implement the McCrary density test. Is there evidence of manipulation?

3. **Placebo test**: Choose a placebo cutoff where no effect should exist. Does the RDD estimate equal zero?

4. **Fuzzy RDD**: Modify the code to handle fuzzy discontinuities (where treatment probability changes but does not jump from 0 to 1).

---

## Projects

### Project 1: RDD Robustness
Conduct a comprehensive RDD analysis with multiple bandwidth choices, polynomial orders, kernel weighting, covariate adjustment, and placebo tests.

### Project 2: Fuzzy RDD
Implement fuzzy RDD using 2SLS. Compare to sharp RDD estimates.
