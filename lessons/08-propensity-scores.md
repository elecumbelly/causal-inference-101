---
title: "Lesson 8: Propensity Score Methods"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 8: Propensity Score Methods

## Opening Story: The Job Training Program

In 1976, the U.S. government launched the National Supported Work Demonstration (NSW), a job training program for disadvantaged workers. The program was randomly assigned in some sites but not others. Paul LaLonde used the NSW data to compare experimental and observational estimates, showing that without randomization, observational methods could produce wildly different results.

The key insight was that without randomization, treated and control groups were systematically different. Propensity score methods offer a way to address this by modeling the probability of treatment and using it to balance the groups.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define propensity scores and their role in causal inference
2. Estimate propensity scores using logistic regression
3. Implement matching, weighting, and stratification
4. Assess covariate balance
5. Use doubly robust estimation

---

```{figure} ../figures/instructional/propensity-overlap.svg
:name: lesson-08-propensity-overlap
:alt: Propensity methods need a region where treated and control units are comparable.
:width: 100%

Propensity methods need a region where treated and control units are comparable.
```

---

## 8.1 The Propensity Score

### Definition

The propensity score $e(X)$ is the probability of treatment given covariates:

$$e(X) = P(T = 1 | X)$$

### The Propensity Score Theorem

Two key results make propensity scores useful:

1. **Dimension reduction**: If exchangeability already holds conditional on the measured covariates $X$, it also holds conditional on the true propensity score $e(X)$
2. **Conditional independence**: Given $e(X)$, treatment is independent of potential outcomes:
   $$(Y(0), Y(1)) \perp T | e(X)$$

### Why This Works

The propensity score is a balancing score for the measured covariates. It summarizes treatment assignment information in $X$; it does **not** balance omitted variables or turn an observational study into a randomized trial.

---

## 8.2 Estimating Propensity Scores

### Logistic Regression

The most common approach:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 1000

# Generate data
age = np.random.normal(40, 10, n)
education = np.random.normal(12, 2, n)
income = np.random.normal(50000, 15000, n)

# Treatment depends on covariates
log_odds = -2 + 0.02 * age + 0.1 * education + 0.00001 * income
prob_treat = 1 / (1 + np.exp(-log_odds))
treatment = np.random.binomial(1, prob_treat)

# Fit propensity score model
X = np.column_stack([age, education, income])
ps_model = LogisticRegression()
ps_model.fit(X, treatment)

# Get propensity scores
propensity_scores = ps_model.predict_proba(X)[:, 1]

print(f"Treatment rate: {treatment.mean():.3f}")
print(f"Mean propensity score: {propensity_scores.mean():.3f}")
print(f"Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
```

---

## 8.3 Propensity Score Matching

### The Idea

Match treated and control units with similar propensity scores, then compare outcomes.

### Types of Matching

1. **Nearest neighbor**: Match each treated unit to the closest control
2. **Caliper matching**: Only match within a specified distance
3. **Mahalanobis matching**: Match on propensity score and key covariates

```python
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(treatment, propensity_scores, caliper=0.1):
    """
    Match treated and control units based on propensity scores.
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))

    matched_pairs = []
    for t_idx in treated_idx:
        dist, c_idx = nn.kneighbors(propensity_scores[t_idx].reshape(1, -1))
        if dist[0][0] < caliper:
            matched_pairs.append((t_idx, control_idx[c_idx[0][0]]))

    return matched_pairs

# Perform matching
matches = propensity_score_matching(treatment, propensity_scores)
print(f"Matched {len(matches)} pairs out of {treatment.sum()} treated units")
```

---

## 8.4 Propensity Score Weighting

### Inverse Probability Weighting (IPW)

Weight each observation by the inverse of its propensity score (for treated) or 1 minus propensity score (for control):

```python
def calculate_ipw_weights(treatment, propensity_scores):
    """
    Calculate inverse probability weights.
    """
    weights = np.where(
        treatment == 1,
        1 / propensity_scores,
        1 / (1 - propensity_scores)
    )
    return weights

weights = calculate_ipw_weights(treatment, propensity_scores)

# Check weight distribution
print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"Weight mean: {weights.mean():.2f}")

# Stabilized weights
stabilized_weights = np.where(
    treatment == 1,
    treatment.mean() / propensity_scores,
    (1 - treatment.mean()) / (1 - propensity_scores)
)

print(f"\nStabilized weight range: [{stabilized_weights.min():.2f}, {stabilized_weights.max():.2f}]")
print(f"Stabilized weight mean: {stabilized_weights.mean():.2f}")
```

---

## 8.5 Assessing Covariate Balance

### The Goal

After weighting or matching, the covariates should be balanced between treated and control groups.

### Standardized Mean Difference (SMD)

```python
def calculate_smd(covariates, treatment, weights=None):
    """
    Calculate standardized mean differences for covariates.
    """
    if weights is None:
        weights = np.ones(len(treatment))

    smds = []
    for i in range(covariates.shape[1]):
        treated_mean = np.average(covariates[treatment == 1, i], weights=weights[treatment == 1])
        control_mean = np.average(covariates[treatment == 0, i], weights=weights[treatment == 0])

        treated_var = np.average((covariates[treatment == 1, i] - treated_mean)**2, weights=weights[treatment == 1])
        control_var = np.average((covariates[treatment == 0, i] - control_mean)**2, weights=weights[treatment == 0])

        pooled_std = np.sqrt((treated_var + control_var) / 2)
        smd = abs(treated_mean - control_mean) / pooled_std
        smds.append(smd)

    return np.array(smds)

# Check balance before and after weighting
smd_before = calculate_smd(X, treatment)
smd_after = calculate_smd(X, treatment, weights)

print("Standardized Mean Differences:")
print("  Before weighting:", smd_before.round(3))
print("  After weighting: ", smd_after.round(3))
print("\nSMD < 0.1 indicates good balance")
```

---

## 8.6 Doubly Robust Estimation

### The Idea

Combine outcome modeling with propensity score weighting for double protection: you're consistent if either the outcome model or propensity model is correct.

### Implementation

```python
def doubly_robust_estimator(outcome, treatment, propensity_scores, covariates):
    """
    Doubly robust estimator for average treatment effect.
    """
    from sklearn.linear_model import LinearRegression

    # Fit outcome model
    treated_mask = treatment == 1
    control_mask = treatment == 0

    model_treated = LinearRegression()
    model_control = LinearRegression()

    model_treated.fit(covariates[treated_mask], outcome[treated_mask])
    model_control.fit(covariates[control_mask], outcome[control_mask])

    # Predict potential outcomes
    y1_hat = model_treated.predict(covariates)
    y0_hat = model_control.predict(covariates)

    # Doubly robust estimator
    dr_term = treatment * (outcome - y1_hat) / propensity_scores - \
              (1 - treatment) * (outcome - y0_hat) / (1 - propensity_scores)

    ate = np.mean(y1_hat - y0_hat + dr_term)
    return ate

# Generate outcome data
outcome = 2 * treatment + 0.5 * age + 0.3 * education + np.random.normal(0, 1, n)

# Estimate ATE
ate_dr = doubly_robust_estimator(outcome, treatment, propensity_scores, X)
print(f"Doubly robust ATE estimate: {ate_dr:.3f}")
```

---

## 8.7 Common Mistakes

1. **Positivity violations**: If some units have propensity scores of 0 or 1, matching fails
2. **Poor balance**: Always check SMD after weighting/matching
3. **Extreme weights**: Stabilize weights to reduce variance
4. **Ignoring model uncertainty**: Use cross-fitting for propensity scores

---

## 8.8 Discussion Questions

1. **Positivity**: What happens when propensity scores are close to 0 or 1?

2. **Weighting vs. Matching**: When might you prefer one over the other?

3. **Double Robustness**: Why is doubly robust estimation preferred?

4. **Balance Assessment**: What else can you check besides SMD?

5. **Model Misspecification**: What happens if your propensity model is wrong?

---

## 8.9 Knowledge Check

### Multiple Choice

1. **The propensity score is:**
   - A) The probability of outcome given treatment
   - B) The probability of treatment given covariates
   - C) The effect of treatment on outcome
   - D) The correlation between treatment and outcome

2. **Propensity scores help with:**
   - A) Confounding
   - B) Selection bias
   - C) Measurement error
   - D) All of the above

3. **SMD < 0.1 indicates:**
   - A) Poor balance
   - B) Good balance
   - C) Perfect balance
   - D) No information

4. **Doubly robust estimation requires:**
   - A) Both models to be correct
   - B) Either model to be correct
   - C) Neither model to be correct
   - D) A specific functional form

5. **Extreme propensity scores cause:**
   - A) High variance
   - B) Low bias
   - C) Perfect balance
   - D) No issues

### Short Answer

6. **Explain why propensity scores are sufficient for confounding adjustment.**

7. **What are the advantages of stabilized weights?**

8. **Describe the role of doubly robust estimation in causal inference.**

9. **How can you assess whether your propensity model is adequate?**

10. **Design a study using propensity score methods for a policy evaluation.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Estimation, overlap, and treatment-effect heterogeneity**](https://www.youtube.com/watch?v=YzcOYU-s2t4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=42)

Places propensity-based estimation in the wider outcome-model and weighting toolkit.

**Active-viewing prompt:** Track where positivity enters and what extreme propensity scores do to variance.
```

---

## 8.10 Summary

1. **Propensity scores** summarize confounding information
2. **Matching, weighting, and stratification** are the main methods
3. **Balance assessment** is crucial for validity
4. **Doubly robust estimation** provides double protection
5. **Positivity** is a key assumption

---

## 8.11 Further Reading

- Rosenbaum, P.R. & Rubin, D.B. (1983). "The Central Role of the Propensity Score in Observational Studies." *Biometrika*.
- Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.


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


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/08-propensity-scores.ipynb)
- [Download the practice lab](../labs/lab08-propensity-scores-practice.ipynb)
- [Download the lab solution](../solutions/lab08-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
