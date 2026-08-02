

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
