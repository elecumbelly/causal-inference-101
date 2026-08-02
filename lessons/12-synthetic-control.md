---
title: "Lesson 12: Synthetic Control Methods"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 12: Synthetic Control Methods

## Opening Story: German Reunification

In 1990, West Germany reunified with East Germany. This created a natural experiment: how did reunification affect the German economy?

Abadie, Diamond, and Hainmueller (2015) used synthetic control methods to construct a "synthetic West Germany" from a weighted combination of other countries that didn't experience reunification. By comparing actual West Germany to its synthetic counterpart, they estimated the causal effect of reunification.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the synthetic control method
2. Construct synthetic controls using optimization
3. Conduct inference with placebo tests
4. Assess the quality of synthetic controls
5. Apply the method to policy evaluation

---

```{figure} ../figures/instructional/synthetic-control.svg
:name: lesson-12-synthetic-control
:alt: A synthetic control should track the treated unit before intervention; the later gap estimates the effect.
:width: 100%

A synthetic control should track the treated unit before intervention; the later gap estimates the effect.
```

---

## 12.1 The Synthetic Control Method

### The Idea

Construct a counterfactual for the treated unit as a weighted combination of untreated units:

$$Y_{it}(0) = \sum_{j=2}^{J+1} w_j Y_{jt}$$

where the weights $w_j$ are chosen to match pre-treatment characteristics and outcomes.

### The Algorithm

1. Choose donor pool (untreated units)
2. Optimize weights to minimize pre-treatment fit
3. Compare post-treatment outcomes

---

## 12.2 Implementation

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Simulated example: 1 treated unit, 10 donor units
n_years = 20
treatment_year = 15
true_effect = -2.0

# Pre-treatment outcomes for treated unit
pre_treatment = np.cumsum(np.random.normal(0.1, 0.5, n_years))

# Donor units (similar trajectories)
donors = np.array([
    pre_treatment + np.random.normal(0, 0.3, n_years) + np.random.normal(0, 0.1) * np.arange(n_years)
    for _ in range(10)
])

# Post-treatment with effect
post_treatment = pre_treatment[-1] + np.cumsum(np.random.normal(0.1, 0.5, n_years - treatment_year))
post_treatment += true_effect  # Add treatment effect

# Combine
y_treated = np.concatenate([pre_treatment, post_treatment])
y_donors = np.column_stack([
    np.column_stack([d[:treatment_year] for d in donors]),
    np.column_stack([d[treatment_year:] for d in donors])
])

# Synthetic control weights (simplified optimization)
from scipy.optimize import minimize

def synthetic_control_weights(y_treated_pre, y_donors_pre):
    """
    Find weights that minimize pre-treatment discrepancy.
    """
    n_donors = y_donors_pre.shape[1]

    def objective(w):
        synthetic = y_donors_pre @ w
        return np.sum((y_treated_pre - synthetic) ** 2)

    # Constraints: weights sum to 1, non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors

    # Initial weights (equal)
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    return result.x

# Get weights
w = synthetic_control_weights(y_treated[:treatment_year], y_donors[:treatment_year])

# Construct synthetic control
y_synthetic = y_donors @ w

# Estimate treatment effect
effect = y_treated[treatment_year:] - y_synthetic[treatment_year:]
print(f"Estimated treatment effect (post-treatment mean): {effect.mean():.3f}")
print(f"True treatment effect: {true_effect}")
```

---

## 12.3 Placebo Tests

```python
# Permutation inference: apply method to each donor unit
effects_placebo = []
for i in range(10):
    # Use donor i as "treated"
    other_donors = np.delete(y_donors, i, axis=1)
    w_placebo = synthetic_control_weights(
        y_donors[:treatment_year, i],
        other_donors[:treatment_year]
    )
    y_synth_placebo = other_donors @ w_placebo
    effect_placebo = y_donors[treatment_year:, i] - y_synth_placebo[treatment_year:]
    effects_placebo.append(effect_placebo.mean())

# Calculate p-value
p_value = np.mean([abs(e) >= abs(effect.mean()) for e in effects_placebo])
print(f"Placebo p-value: {p_value:.3f}")
```

---

## 12.4 Common Mistakes

1. **Poor pre-treatment fit**: Check RMSPE
2. **Spillovers**: Ensure donor units aren't affected
3. **Overfitting**: Use cross-validation for weight selection
4. **Inference**: Always conduct placebo tests

---

## 12.5 Knowledge Check

### Multiple Choice

1. **Synthetic control constructs:**
   - A) A randomized experiment
   - B) A counterfactual from weighted donors
   - C) A difference-in-differences estimate
   - D) An instrumental variable

2. **Weights are chosen to:**
   A) Maximize treatment effect
   B) Minimize pre-treatment discrepancy
   C) Be equal
   D) Be binary

3. **Placebo tests assess:**
   A) Statistical significance
   B) Practical significance
   C) Both A and B
   D) Neither

4. **The donor pool should:**
   A) Include the treated unit
   B) Not be affected by treatment
   C) Be as large as possible
   D) Be similar to the treated unit

5. **Synthetic control is useful when:**
   A) There's one treated unit
   B) There's randomization
   C) There's a clear cutoff
   D) There are many treated units

### Short Answer

6. **Explain how synthetic control differs from matching.**

7. **Why are placebo tests important?**

8. **How do you assess the quality of a synthetic control?**

9. **What is the role of pre-treatment fit?**

10. **Give an example where synthetic control would be appropriate.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Alberto Abadie on synthetic control**](https://www.youtube.com/watch?v=nKzNp-qpE-I&list=PLoazKTcS0RzZ1SUgeOgc6SWt51gfT80N0&index=11)

A lecture from the method’s leading originator on constructing and diagnosing synthetic controls.

**Active-viewing prompt:** Look for the role of pre-treatment fit and why placebo gaps matter.
```

---

## 12.6 Summary

1. **Synthetic control** constructs counterfactuals from weighted donors
2. **Pre-treatment fit** is crucial for validity
3. **Placebo tests** provide inference
4. **Single-unit studies** can be credible with this method
5. **Donor pool selection** affects results

---

## 12.7 Further Reading

- Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and the Synthetic Control Method." *American Journal of Political Science*.
- Abadie, A. (2021). "Using Synthetic Control: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*.


---

## Worked Examples

### Example 1: Economics — California Tobacco Control (Abadie et al. 2010)

California implemented Proposition 99 (tobacco tax) in 1988. The synthetic control is a weighted combination of other states that best matches California's pre-treatment cigarette consumption trends.

### Example 2: Policy — German Reunification (Abadie et al. 2015)

West Germany reunified with East Germany in 1990. The synthetic control is a weighted combination of OECD countries that best matches West Germany's pre-reunification GDP trends.

### Example 3: Technology — EU GDPR

The EU implemented GDPR in 2018. A synthetic EU is constructed from non-EU countries to estimate GDPR's effect on tech innovation.

### Example 4: Health — Oregon Health Insurance Experiment

Oregon expanded Medicaid via lottery. The synthetic control method constructs a comparison for lottery winners using non-winners.

---

## Diagnostics: Synthetic Control Quality

### Pre-Treatment Fit Assessment

```python
import numpy as np

def assess_synthetic_control(pre_treatment_actual, pre_treatment_synthetic):
    """Assess quality of synthetic control match."""

    rmspe = np.sqrt(np.mean((pre_treatment_actual - pre_treatment_synthetic) ** 2))
    mape = np.mean(np.abs(pre_treatment_actual - pre_treatment_synthetic) /
                   np.abs(pre_treatment_actual)) * 100
    corr = np.corrcoef(pre_treatment_actual, pre_treatment_synthetic)[0, 1]

    print(f"Pre-Treatment Fit Quality:")
    print(f"  RMSPE: {rmspe:.3f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Correlation: {corr:.3f}")

    return rmspe, mape, corr
```

### Permutation Tests

```python
def permutation_test_sc(treated_outcome, synthetic_outcome, donor_outcomes):
    """Permutation test for synthetic control significance."""

    gap_treated = treated_outcome - synthetic_outcome

    placebo_gaps = []
    for donor in donor_outcomes:
        placebo_gaps.append(donor['actual'] - donor['synthetic'])

    all_gaps = np.array([gap_treated] + placebo_gaps)
    p_value = np.mean(np.abs(all_gaps[1:]) >= np.abs(gap_treated))

    print(f"Permutation test p-value: {p_value:.3f}")

    return p_value
```

---

## Interpretation Workshop

### Reading Synthetic Control Studies

Key questions:
1. **How was the synthetic control constructed?** What variables were used for matching?
2. **How well does it fit pre-treatment outcomes?** (RMSPE, visual inspection)
3. **What donors were included?** Are they comparable?
4. **Is the post-treatment divergence statistically significant?** (Permutation test)
5. **Are there robustness checks?** (Leave-one-out, alternative donors)

### The Role of Donor Pool

The synthetic control is only as good as the donor pool:
- Too few donors: Limited flexibility to match pre-treatment trends
- Too many donors: Risk of overfitting
- Similar donors: Better matches, but may share the treatment

---

## Practical Application

### Implementing Synthetic Control

```python
import numpy as np
from scipy.optimize import minimize

class SyntheticControl:
    """Synthetic control method implementation."""

    def __init__(self, treated_unit, donor_units):
        self.treated_unit = treated_unit
        self.donor_units = donor_units
        self.weights = None

    def fit(self, treated_pre, donor_pre):
        """Find optimal weights for synthetic control."""

        n_donors = len(self.donor_units)

        def objective(weights):
            synthetic = donor_pre @ weights
            return np.sqrt(np.mean((treated_pre - synthetic) ** 2))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_donors)]
        w0 = np.ones(n_donors) / n_donors

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x

        print("Synthetic Control Weights:")
        for unit, weight in zip(self.donor_units, self.weights):
            if weight > 0.01:
                print(f"  {unit}: {weight:.3f}")

        return self

    def predict(self, donor_data):
        """Predict synthetic control."""
        return donor_data @ self.weights

    def estimate_effect(self, treated_post, synthetic_post):
        """Estimate treatment effect."""
        effect = np.mean(treated_post - synthetic_post)
        print(f"Treatment Effect: {effect:.3f}")
        return effect
```

---

## Limitations

- **Single treated unit**: Method designed for N=1 case studies
- **Pre-treatment fit**: No guarantee of good pre-treatment match
- **Extrapolation**: Synthetic control may not represent counterfactual
- **Inference**: Permutation tests may have low power with few donors

---

## Exercises

1. **Implement synthetic control**: Use the code to construct a synthetic control for a treated unit. How well does it match pre-treatment outcomes?

2. **Donor pool sensitivity**: Remove one donor at a time and re-estimate. How sensitive are results to the donor pool?

3. **Covariate selection**: What happens when you include different covariates in the matching? How do weights change?

4. **Inference**: Conduct a permutation test. Is the post-treatment effect significant?

---

## Projects

### Project 1: Synthetic Control Application
Apply synthetic control to a policy intervention of your choice. Select treated unit and donor pool, construct synthetic control, assess pre-treatment fit, estimate treatment effect, and conduct robustness checks.

### Project 2: Augmented Synthetic Control
Implement the augmented synthetic control method (Ben-Michael et al. 2021) that combines synthetic control with outcome modeling.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/12-synthetic-control.ipynb)
- [Download the practice lab](../labs/lab12-synthetic-control-practice.ipynb)
- [Download the lab solution](../solutions/lab12-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
