

---

## Worked Examples

### Example 1: Medicine — Trial to Clinical Practice

A drug works in a randomized trial with strict inclusion criteria. But in clinical practice, patients are older, sicker, and have more comorbidities. The treatment effect may not transport to this broader population.

### Example 2: Economics — Training Program Generalizability

A job training program shows positive effects in urban areas. Will it work in rural areas? Differences in labor markets, demographics, and program implementation may limit transportability.

### Example 3: Technology — Platform A/B Test to Global Rollout

A feature tested on US users may not work the same way in other cultures. External validity requires understanding how context affects treatment effects.

### Example 4: Policy — Pilot Program to National Scale

A pilot program shows positive results in one city. Scaling nationally introduces new challenges: different administrative capacities, population compositions, and implementation quality.

---

## Diagnostics: Assessing External Validity

### Transportability Conditions

For causal effects to transport from study population to target population:
1. **Same causal structure**: The DAG must be the same in both populations
2. **No effect modification by population**: The CATE function must be the same
3. **Representativeness**: The study sample must cover the range of the target population

```python
import numpy as np
import pandas as pd

def check_transportability(df_study, df_target, treatment, outcome, covariates):
    """Check if causal effects can be transported."""

    # Check covariate overlap
    print("Covariate Distribution Comparison:")
    print("=" * 50)

    for cov in covariates:
        study_mean = df_study[cov].mean()
        target_mean = df_target[cov].mean()
        diff = abs(study_mean - target_mean) / df_study[cov].std()

        status = "OK" if diff < 0.5 else "WARN"
        print(f"  {cov}: Study={study_mean:.2f}, Target={target_mean:.2f}, SMD={diff:.2f} [{status}]")

    # Check outcome model stability
    from sklearn.linear_model import LinearRegression

    model_study = LinearRegression().fit(df_study[covariates], df_study[outcome])
    model_target = LinearRegression().fit(df_target[covariates], df_target[outcome])

    r_sq_study = model_study.score(df_study[covariates], df_study[outcome])
    r_sq_target = model_target.score(df_target[covariates], df_target[outcome])

    print(f"\nOutcome model R-squared:")
    print(f"  Study: {r_sq_study:.3f}")
    print(f"  Target: {r_sq_target:.3f}")
```

### Sensitivity Analysis for Transportability

```python
def transport_sensitivity(ate_study, sensitivity_params):
    """Assess sensitivity of transported effect to violations."""

    results = {}

    for param_name, param_value in sensitivity_params.items():
        transported_effect = ate_study * param_value
        results[param_name] = transported_effect
        print(f"  {param_name}: {transported_effect:.3f}")

    return results
```

---

## Interpretation Workshop

### When External Validity Fails

- **Different populations**: Trial participants differ from target population
- **Different contexts**: Treatment delivery differs between study and practice
- **Different time periods**: Effects may change over time
- **Hawthorne effects**: Study participation itself affects behavior

### Reporting External Validity

When presenting results:
1. Clearly describe the study population
2. Compare to the target population on observables
3. Discuss potential effect modifiers
4. Present sensitivity analyses for transportability

---

## Practical Application

### Improving External Validity

1. **Inclusive trial design**: Broaden inclusion criteria
2. **Pragmatic trials**: Conduct trials in real-world settings
3. **Stratified analysis**: Report effects by subgroup
4. **Replication**: Conduct the study in multiple settings
5. **Meta-analysis**: Combine results across diverse studies

### Weighting for Transportability

```python
def transport_weights(df_study, df_target, covariates):
    """Compute weights to transport study to target population."""

    from sklearn.linear_model import LogisticRegression

    # Combine datasets
    df_study = df_study.copy()
    df_study['population'] = 1  # Study

    df_target = df_target.copy()
    df_target['population'] = 0  # Target

    df_combined = pd.concat([df_study, df_target])

    # Fit propensity model
    model = LogisticRegression(max_iter=1000)
    model.fit(df_combined[covariates], df_combined['population'])

    # Get weights for study population
    probs = model.predict_proba(df_study[covariates])[:, 1]
    weights = (1 - probs) / probs  # Target vs study odds

    return weights / np.mean(weights)  # Normalize
```

---

## Limitations

- Cannot test transportability assumptions empirically
- Effect modification may be unmeasured
- Weights can be unstable with poor covariate overlap
- Context-specific factors may be unobservable

---

## Exercises

1. **Transportability check**: Compare a clinical trial population to a real-world population. What are the key differences?
2. **Weighting**: Implement transport weighting. How do the results change?
3. **Sensitivity analysis**: How much effect modification would need to exist to invalidate transport?
4. **Critique**: Find a study that claims generalizability. Evaluate the evidence.

---

## Projects

### Project 1: Transportability Analysis
Conduct a transportability analysis moving causal findings from a trial to a real-world population.

### Project 2: External Validity Assessment
Assess the external validity of a published RCT by comparing the study population to the target population.
