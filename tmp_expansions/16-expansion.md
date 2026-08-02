

---

## Worked Examples

### Example 1: Medicine — Dynamic Treatment Regimes

A patient receives treatment based on their evolving health status. Marginal structural models estimate the effect of this dynamic treatment strategy by properly handling time-varying confounding.

### Example 2: Economics — Job Training with Time-Varying Eligibility

Workers become eligible for training based on employment history. Inverse probability of treatment weighting (IPTW) adjusts for time-varying confounding by reweighting observations.

### Example 3: Technology — Adaptive A/B Testing

A platform dynamically adjusts feature exposure based on user behavior. G-computation estimates the effect of the adaptive strategy by simulating counterfactual histories.

### Example 4: Policy — Vaccination Campaigns

Vaccination rates change over time based on disease prevalence. Time-varying methods account for this feedback loop between outcome and treatment.

---

## Diagnostics: G-methods

### Inverse Probability of Treatment Weighting

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def iptw_weights(df, treatment_cols, confounder_cols):
    """Calculate IPTW weights for time-varying treatment."""

    weights = np.ones(len(df))

    for t in range(len(treatment_cols)):
        treatment = df[treatment_cols[t]]
        confounders = df[confounder_cols[:t+1]]

        model = LogisticRegression(max_iter=1000)
        model.fit(confounders, treatment)
        prob_treatment = model.predict_proba(confounders)[:, 1]

        weight_t = np.where(treatment == 1,
                           1 / prob_treatment,
                           1 / (1 - prob_treatment))
        weights *= weight_t

    return weights
```

### G-computation

```python
def g_computation(df, treatment_history, outcome, confounders):
    """G-computation for dynamic treatment regimes."""

    from sklearn.ensemble import GradientBoostingRegressor

    X = pd.concat([treatment_history, confounders], axis=1)
    model = GradientBoostingRegressor()
    model.fit(X, outcome)

    # Predict under different treatment regimes
    X_always_treat = X.copy()
    X_always_treat[treatment_history.columns] = 1
    y_always_treat = model.predict(X_always_treat).mean()

    X_never_treat = X.copy()
    X_never_treat[treatment_history.columns] = 0
    y_never_treat = model.predict(X_never_treat).mean()

    effect = y_always_treat - y_never_treat

    print(f"Effect of always-treat vs never-treat: {effect:.3f}")

    return effect
```

---

## Interpretation Workshop

### When to Use G-methods

- Time-varying confounding affected by prior treatment
- Dynamic treatment regimes
- Joint treatment-confounder feedback

### Key Assumptions

1. **Sequential ignorability**: No unmeasured confounders at each time point
2. **Correct model specification**: Treatment and outcome models are correct
3. **Positivity**: Non-zero probability of treatment at each time point

---

## Practical Application

### Marginal Structural Models

```python
import statsmodels.api as sm

def marginal_structural_model(df, treatment_col, outcome_col, weight_col):
    """Estimate MSM using IPTW."""

    X = sm.add_constant(df[[treatment_col]])
    weights = df[weight_col]

    model = sm.WLS(df[outcome_col], X, weights=weights).fit()

    effect = model.params[treatment_col]
    se = model.bse[treatment_col]

    print(f"MSM Estimate: {effect:.3f} (SE: {se:.3f})")

    return model
```

---

## Limitations

- Weight instability: Extreme weights can cause variance inflation
- Positivity violations: Near-deterministic treatment assignment
- Model misspecification: Incorrect treatment or outcome models
- Data requirements: Need longitudinal data with time-varying measures

---

## Exercises

1. **IPTW implementation**: Calculate IPTW weights for a time-varying treatment. How do weights behave over time?
2. **Weight truncation**: Compare results with and without weight truncation. When does truncation help?
3. **G-computation**: Implement G-computation for a dynamic treatment regime. Compare to IPTW results.
4. **Critique**: Find a study using G-methods. Evaluate the sequential ignorability assumption.

---

## Projects

### Project 1: G-methods Comparison
Compare IPTW, TMLE, and G-computation on simulated data with time-varying confounding.

### Project 2: Dynamic Treatment Regime
Estimate the optimal dynamic treatment regime using Q-learning and compare to standard-of-care.
