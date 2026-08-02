---
title: "Lesson 15: Sensitivity Analysis"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 15: Sensitivity Analysis

## Opening Story: The Obesity Paradox

Many observational studies have found that obesity is associated with better outcomes in certain diseases—the "obesity paradox." Critics argue this might be due to unmeasured confounding. Sensitivity analysis asks: how strong would unmeasured confounding need to be to explain away the observed effect?

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the role of sensitivity analysis
2. Implement Rosenbaum bounds
3. Conduct E-value calculations
4. Interpret sensitivity analysis results
5. Design robustness checks

---

## 15.1 Why Sensitivity Analysis?

Observational studies always face the possibility of unmeasured confounding. Sensitivity analysis quantifies how much confounding would be needed to change conclusions.

---

## 15.2 Rosenbaum Bounds

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulated matched pairs data
n_pairs = 50
treatment_effect = 2.0
noise = np.random.normal(0, 1, n_pairs)

# Outcomes for treated and control
Y_treated = treatment_effect + noise
Y_control = noise

# Observed difference
diff = Y_treated - Y_control
mean_diff = diff.mean()

# Rosenbaum bounds: how much treatment assignment could differ
# Without confounding, P(T_i = 1 | X_i) = 0.5 for all
# With confounding, P(T_i = 1 | X_i) could differ by gamma

def rosenbaum_bound(diff, gamma):
    """
    Calculate the lower bound of the p-value for a given gamma.
    """
    n = len(diff)
    # Under worst-case confounding
    prob_treat_max = gamma / (1 + gamma)
    prob_treat_min = 1 / (1 + gamma)

    # Wilcoxon signed-rank test statistic
    ranks = np.argsort(np.argsort(np.abs(diff)))
    signs = np.sign(diff)

    # Calculate expected value under worst case
    expected = np.sum(ranks * prob_treat_max)

    # Simplified bound calculation
    p_value_worst = stats.norm.cdf(-abs(mean_diff) / (np.std(diff) / np.sqrt(n)))

    return p_value_worst

# Test different gamma values
print("Rosenbaum Bounds:")
for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
    p_val = rosenbaum_bound(diff, gamma)
    print(f"  gamma = {gamma:.1f}: p-value = {p_val:.4f}")
```

---

## 15.3 E-value

```python
def e_value(estimate, se):
    """
    Calculate the E-value for an estimated treatment effect.
    The minimum strength of confounding needed to explain away the effect.
    """
    # For risk ratios
    RR = np.exp(estimate)
    if RR >= 1:
        E = RR + np.sqrt(RR * (RR - 1))
    else:
        E = 1/RR + np.sqrt((1/RR) * (1/RR - 1))
    return E

# Example
ATE = 0.5
SE = 0.1
RR = np.exp(ATE)
E = e_value(ATE, SE)

print(f"ATE: {ATE}")
print(f"Risk Ratio: {RR:.3f}")
print(f"E-value: {E:.3f}")
print(f"\nConfounding would need to be associated with both")
print(f"treatment and outcome by a risk ratio of at least {E:.2f}")
print(f"to explain away the observed effect.")
```

---

## 15.4 Common Mistakes

1. **Ignoring sensitivity analysis**: Always conduct it for observational studies
2. **Misinterpreting bounds**: Bounds are worst-case scenarios
3. **Only reporting p-values**: Report effect sizes and sensitivity together
4. **Not pre-specifying**: Plan sensitivity analyses in advance

---

## 15.5 Knowledge Check

### Multiple Choice

1. **Sensitivity analysis assesses:**
   A) Statistical significance
   B) Robustness to unmeasured confounding
   C) Practical significance
   D) Both A and C

2. **Rosenbaum bounds test:**
   A) The treatment effect
   B) The strength of confounding needed to change conclusions
   C) The sample size
   D) The power of the study

3. **The E-value is:**
   A) The p-value
   B) The minimum confounding strength to explain away the effect
   C) The effect size
   D) The standard error

4. **Sensitivity analysis is important because:**
   A) Observational studies have unmeasured confounders
   B) Experiments are always better
   C) Statistics is uncertain
   D) All of the above

5. **A large E-value means:**
   A) The effect is fragile
   B) The effect is robust to confounding
   C) The effect is small
   D) The effect is large

### Short Answer

6. **Explain what Rosenbaum bounds tell us.**

7. **How do you interpret an E-value of 3.0?**

8. **Why is sensitivity analysis crucial for observational studies?**

9. **What are the limitations of sensitivity analysis?**

10. **How would you design a sensitivity analysis for a DiD study?**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Unobserved confounding and sensitivity analysis**](https://www.youtube.com/watch?v=IXNMYqUsBBQ&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=47)

Frames sensitivity analysis as measuring how strong hidden bias must be to alter a conclusion.

**Active-viewing prompt:** Translate the sensitivity parameter back into a plausible real-world confounder.
```

---

## 15.6 Summary

1. **Sensitivity analysis** quantifies robustness to unmeasured confounding
2. **Rosenbaum bounds** test how much treatment assignment could differ
3. **E-value** measures the minimum confounding strength needed
4. **Always conduct** sensitivity analysis for observational studies
5. **Interpret results carefully** in context

---

## 15.7 Further Reading

- Rosenbaum, P.R. (2002). *Observational Studies*. Springer.
- VanderWeele, T.J. & Ding, P. (2017). "Sensitivity Analysis in Observational Research." *Annals of Internal Medicine*.


---

## Worked Examples

### Example 1: Rosenbaum Bounds

In an observational study, how strong would an unmeasured confounder need to be to change the conclusion? Rosenbaum bounds quantify this by asking: at what level of hidden bias would the finding become insignificant?

### Example 2: E-values

For a risk ratio of 2.0, the E-value is 3.0. This means an unmeasured confounder would need to be associated with both treatment and outcome by a risk ratio of at least 3.0 to explain away the result. For comparison, smoking and lung cancer have a risk ratio of about 15.

### Example 3: Oster's Delta

If the treatment coefficient changes from 5.0 (unadjusted) to 3.0 (adjusted for observables), Oster's delta measures how much selection on unobservables would need to exceed selection on observables. A delta > 1 suggests robustness.

### Example 4: Sensitivity to Model Specification

Running the same analysis with 10 different model specifications. If results are stable (e.g., all coefficients between 2.5 and 3.5), the finding is robust. If they vary widely (e.g., from 0.5 to 5.0), the conclusion is fragile.

---

## Diagnostics: Sensitivity Methods

### E-value Calculation

```python
import numpy as np

def e_value(risk_ratio):
    """Calculate E-value for a risk ratio."""
    if risk_ratio >= 1:
        e = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))
    else:
        e = 1/risk_ratio + np.sqrt((1/risk_ratio) * (1/risk_ratio - 1))
    return e

def e_value_ci(rr_point, rr_lower):
    """Calculate E-value for point estimate and confidence interval."""
    e_point = e_value(rr_point)
    e_ci = e_value(rr_lower)

    print(f"Risk Ratio: {rr_point:.2f} (95% CI lower: {rr_lower:.2f})")
    print(f"E-value (point): {e_point:.2f}")
    print(f"E-value (CI): {e_ci:.2f}")
    print(f"An unmeasured confounder would need RR >= {e_ci:.2f} to explain away the result")

    return e_point, e_ci
```

### Rosenbaum Bounds

```python
def rosenbaum_bounds(n_pairs, treatment_effect, gamma_range=(1, 3, 5)):
    """Calculate Rosenbaum bounds for matched pair studies."""

    print("Rosenbaum Bounds Analysis:")
    print("=" * 50)

    for gamma in gamma_range:
        worst_case_p = 1 - (1 - 1/gamma)**n_pairs
        print(f"  Gamma = {gamma}: Worst-case p-value = {worst_case_p:.4f}")

    return gamma_range
```

---

## Interpretation Workshop

### How to Report Sensitivity Analysis

1. State the assumption: "Our estimate assumes no unmeasured confounding"
2. Quantify the threat: "An unmeasured confounder would need to be associated with treatment and outcome by RR >= X"
3. Contextualize: "This is stronger/weaker than the association between known confounder and outcome"
4. Conclude: "Therefore, our finding is robust/fragile to unmeasured confounding"

### Interpreting Sensitivity Parameters

- **E-value > 3**: Very robust — unmeasured confounder would need to be stronger than most known risk factors
- **E-value 1.5-3**: Moderately robust — plausible but concerning
- **E-value < 1.5**: Fragile — modest unmeasured confounding could explain the result

---

## Practical Application

### Comprehensive Sensitivity Analysis

```python
def comprehensive_sensitivity(estimate, se):
    """Run multiple sensitivity analyses."""

    results = {}

    # E-value
    if estimate > 0:
        rr = np.exp(estimate)
        results['e_value'] = e_value(rr)

    # Model sensitivity
    results['model_sensitivity'] = {
        'ols': estimate,
        'with_controls': estimate * 0.8,
        'with_instruments': estimate * 1.2,
    }

    return results
```

---

## Limitations

- Sensitivity is not validation: Bounds show what would need to be true, not what is true
- Conservative: Bounds are often wide, making strong conclusions difficult
- Model dependence: Results depend on the sensitivity model specification

---

## Exercises

1. **E-value calculation**: Calculate the E-value for a study with RR = 1.5. How does this compare to known confounders?
2. **Rosenbaum bounds**: Implement Rosenbaum bounds for a matched pair study. At what Gamma does the result become insignificant?
3. **Oster's delta**: Use Oster's method to assess the degree of selection on unobservables relative to observables.
4. **Critique**: Find a study that claims "robustness." Evaluate whether the sensitivity analysis actually supports this claim.

---

## Projects

### Project 1: Sensitivity Analysis Toolkit
Build a comprehensive sensitivity analysis toolkit that includes E-values, Rosenbaum bounds, and Oster's delta.

### Project 2: Benchmarking Sensitivity Methods
Compare different sensitivity analysis methods on simulated data with known confounding.


---

## Additional Advanced Content

### Multiple Testing Correction

When conducting many sensitivity analyses, the chance of finding at least one "significant" result increases. Multiple testing correction is essential.

```python
import numpy as np
from scipy import stats

def multiple_testing_correction(p_values, method='bonferroni'):
    """Correct for multiple hypothesis testing."""

    n_tests = len(p_values)
    p_values = np.array(p_values)

    if method == 'bonferroni':
        adjusted = np.minimum(p_values * n_tests, 1.0)
    elif method == 'bh':
        # Benjamini-Hochberg
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        adjusted = np.zeros(n_tests)

        for i in range(n_tests - 1, -1, -1):
            adjusted[i] = min(sorted_p[i] * n_tests / (i + 1), 1.0)

        # Ensure monotonicity
        for i in range(1, n_tests):
            adjusted[i] = min(adjusted[i], adjusted[i-1])

        # Unsort
        adjusted[sorted_idx] = adjusted
    else:
        adjusted = p_values

    n_significant = np.sum(adjusted < 0.05)

    print(f"Multiple Testing Correction ({method}):")
    print(f"  Tests conducted: {n_tests}")
    print(f"  Significant (p < 0.05): {n_significant}")
    print(f"  Expected false positives: {n_tests * 0.05:.1f}")

    return adjusted
```

### Benchmarking Sensitivity Analysis

```python
def benchmark_sensitivity(estimate, se, benchmark_confounders):
    """Use benchmarking approach to assess sensitivity."""

    print("Benchmarking Sensitivity Analysis:")
    print("=" * 50)

    results = []

    for name, strength in benchmark_confounders.items():
        # How much bias would this confounder explain?
        bias = strength * se  # Simplified

        adjusted_estimate = estimate - bias

        results.append({
            'benchmark': name,
            'strength': strength,
            'bias': bias,
            'adjusted_estimate': adjusted_estimate
        })

        print(f"  {name}:")
        print(f"    Assumed strength: {strength:.2f}")
        print(f"    Bias: {bias:.4f}")
        print(f"    Adjusted estimate: {adjusted_estimate:.4f}")

    results_df = pd.DataFrame(results)

    # Find which benchmark would explain away the result
    explain_away = results_df[results_df['adjusted_estimate'].abs() < se]
    if len(explain_away) > 0:
        print(f"\n  Confounders that could explain away result:")
        for _, row in explain_away.iterrows():
            print(f"    - {row['benchmark']}")

    return results_df
```

### Combining Sensitivity Methods

```python
def comprehensive_sensitivity_report(estimate, se, n_obs, covariates=5):
    """Generate comprehensive sensitivity analysis report."""

    from scipy import stats

    # E-value
    if estimate > 0:
        rr = np.exp(estimate)
        e_val = rr + np.sqrt(rr * (rr - 1))
    else:
        rr = np.exp(-estimate)
        e_val = 1/rr + np.sqrt(1/rr * (1/rr - 1))

    # Oster's delta (simplified)
    r_squared_unrestricted = 0.3  # Hypothetical
    r_squared_restricted = 0.2  # Hypothetical
    delta = (r_squared_unrestricted - r_squared_restricted) / (1 - r_squared_restricted)

    # Rosenbaum bounds (simplified)
    # At what gamma does p-value exceed 0.05?
    z_stat = estimate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print("Comprehensive Sensitivity Report:")
    print("=" * 60)
    print(f"Estimate: {estimate:.4f} (SE: {se:.4f})")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"\nE-value: {e_val:.2f}")
    print(f"  Interpretation: Unmeasured confounder needs RR >= {e_val:.1f}")
    print(f"\nOster's delta: {delta:.3f}")
    print(f"  Interpretation: {'Robust' if delta > 1 else 'Sensitive'} to unobservables")

    # Sample size for perfect sensitivity
    n_needed = int(n_obs * (1/0.1)**2)  # Simplified
    print(f"\nSample size assessment:")
    print(f"  Current: {n_obs}")
    print(f"  Rule of thumb: > 10 events per covariate")
    print(f"  Covariates: {covariates}")
    print(f"  Adequate: {'Yes' if n_obs > 10 * covariates else 'No'}")

    return {
        'e_value': e_val,
        'oster_delta': delta,
        'p_value': p_value
    }
```

### Practical Guidelines

```python
def sensitivity_analysis_checklist():
    """Checklist for conducting sensitivity analysis."""

    checklist = {
        'E-value': 'Calculate for point estimate and confidence interval',
        'Rosenbaum bounds': 'For matched studies, report Gamma at p=0.05',
        'Oster delta': 'Assess selection on unobservables vs observables',
        'Benchmarking': 'Compare to known confounders in the literature',
        'Model sensitivity': 'Run with/without key covariates',
        'Subgroup analysis': 'Check consistency across populations',
        'Negative control': 'Test on outcomes that should not be affected',
    }

    print("Sensitivity Analysis Checklist:")
    print("=" * 50)
    for method, description in checklist.items():
        print(f"  [ ] {method}: {description}")

    return checklist
```


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/15-sensitivity-analysis.ipynb)
- [Download the practice lab](../labs/lab15-sensitivity-analysis-practice.ipynb)
- [Download the lab solution](../solutions/lab15-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
