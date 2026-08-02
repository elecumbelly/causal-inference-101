

---

## Worked Examples

### Example 1: Medicine — Observational Drug Comparison

Compare outcomes for patients who received Drug A vs Drug B in an observational study. Patients choosing Drug A tend to be sicker (doctors prescribe it to severe cases). Propensity score matching creates comparable groups by matching treated and control patients with similar propensity scores.

### Example 2: Economics — Effect of Marriage on Earnings

Married men earn more than unmarried men. But marriage correlates with traits that also affect earnings (responsibility, stability). Propensity score methods create comparable married and unmarried groups, estimating the effect of marriage net of these confounders.

### Example 3: Technology — Feature Adoption

A platform wants to know if a new feature increases retention. Early adopters of the feature differ from non-adopters in important ways. Propensity score weighting adjusts for these differences by reweighting the sample.

### Example 4: Policy — Training Program Evaluation

A job training program targets disadvantaged workers. Propensity score methods create a comparison group of similar workers who did not participate, estimating what would have happened to participants absent the program.

---

## Diagnostics: Propensity Score Quality

### Common Support Assessment

```python
import numpy as np
import matplotlib.pyplot as plt

def check_common_support(ps_treatment, ps_control):
    """Assess overlap in propensity score distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(ps_treatment, alpha=0.5, label='Treatment', bins=30, density=True)
    ax.hist(ps_control, alpha=0.5, label='Control', bins=30, density=True)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distribution')
    ax.legend()

    # Check overlap
    min_overlap = max(ps_treatment.min(), ps_control.min())
    max_overlap = min(ps_treatment.max(), ps_control.max())
    print(f"Common support range: [{min_overlap:.3f}, {max_overlap:.3f}]")

    return fig
```

### Balance Statistics

After matching or weighting, check:
1. **Standardized mean differences** (SMD): Should be < 0.1
2. **Variance ratios**: Should be close to 1
3. **Visual overlap**: Distributions should overlap substantially

---

## Interpretation Workshop

### Comparing Methods

| Method | Pros | Cons |
|--------|------|------|
| **Matching** | Intuitive, transparent | Drops non-overlapping units |
| **Weighting** | Uses all data | Sensitive to extreme weights |
| **Stratification** | Simple to implement | Assumes constant effect within strata |
| **Regression adjustment** | Efficient | Model-dependent |

### Reading Propensity Score Studies

Key questions to ask:
1. What covariates were used to estimate the propensity score?
2. How was model fit assessed?
3. Was common support enforced? How many units were dropped?
4. Was sensitivity analysis conducted for unmeasured confounding?

---

## Practical Application

### Step-by-Step Propensity Score Analysis

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def propensity_score_matching(df, treatment_col, covariate_cols, n_neighbors=1):
    """Complete propensity score matching workflow."""

    # Step 1: Estimate propensity scores
    X = df[covariate_cols]
    treatment = df[treatment_col]

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, treatment)
    df = df.copy()
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]

    # Step 2: Check common support
    treated = df[treatment == 1]['propensity_score']
    control = df[treatment == 0]['propensity_score']

    min_ps = max(treated.min(), control.min())
    max_ps = min(treated.max(), control.max())

    df_trimmed = df[(df['propensity_score'] >= min_ps) &
                     (df['propensity_score'] <= max_ps)]

    # Step 3: Match and assess balance
    treated_df = df_trimmed[df_trimmed[treatment_col] == 1]
    control_df = df_trimmed[df_trimmed[treatment_col] == 0]

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(control_df[['propensity_score']])
    distances, indices = nn.kneighbors(treated_df[['propensity_score']])

    # Step 4: Report balance
    for col in covariate_cols:
        smd = abs(treated_df[col].mean() - control_df[col].mean()) / treated_df[col].std()
        status = "OK" if smd < 0.1 else "WARN"
        print(f"  {col}: SMD = {smd:.3f} [{status}]")

    return df_trimmed
```

---

## Limitations

- Only adjusts for observed confounders
- Model dependence: Results depend on the propensity score model specification
- Extrapolation: May match units that are fundamentally different
- Balance does not equal causality: Even perfect balance does not guarantee causal identification

---

## Exercises

1. **Method comparison**: Apply matching, weighting, and stratification to the same dataset. Compare results and discuss why they differ.

2. **Sensitivity analysis**: Use Rosenbaum bounds to assess how strong an unmeasured confounder would need to be to explain away the treatment effect.

3. **Diagnostics**: Write code to generate a Love plot showing covariate balance before and after matching.

4. **Application**: Use propensity scores to estimate the effect of smoking on lung cancer, adjusting for age, sex, and occupation.

---

## Projects

### Project 1: Propensity Score Sensitivity Analysis
Create a simulation study that generates data with known causal effect, estimates the effect using propensity scores with different model specifications, and shows how results vary with model choice.

### Project 2: Propensity Score Methods Comparison
Compare propensity score methods (matching, weighting, stratification, regression adjustment) on bias, variance, coverage of confidence intervals, and robustness to model misspecification.
