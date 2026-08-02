

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
