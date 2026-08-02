

---

## Worked Examples

### Example 1: Economics — Minimum Wage Study (Card & Krueger 1994)

New Jersey raised its minimum wage; Pennsylvania did not. Fast-food employment was measured before and after the change.

**DiD estimate**: (NJ_after - NJ_before) - (PA_after - PA_before)

**Key insight**: The parallel trends assumption means NJ employment would have followed the same trend as PA absent the policy change.

### Example 2: Policy — Seat Belt Law

California enacted a mandatory seat belt law. Neighboring states did not. Traffic fatalities were measured before and after.

**Challenge**: California may have had different underlying trends in traffic safety due to other concurrent policies.

### Example 3: Technology — Platform Policy Change

A social media platform changes its algorithm in the US but not in Europe. User engagement is measured before and after.

**SUTVA violation**: Users in different countries interact, potentially contaminating the control group.

### Example 4: Health — Hospital Quality Initiative

A hospital network implements a quality improvement program. Non-participating hospitals serve as controls. Staggered adoption requires modern DiD methods.

---

## Diagnostics: Parallel Trends

### Testing Parallel Trends

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_parallel_trends(df, unit_col, time_col, outcome_col, treatment_time, treatment_col):
    """Test parallel trends assumption using pre-treatment periods."""

    pre_data = df[df[time_col] < treatment_time]
    pre_periods = sorted(pre_data[time_col].unique())

    results = []
    for t in pre_periods:
        period_data = pre_data[pre_data[time_col] == t]
        treated_mean = period_data[period_data[treatment_col] == 1][outcome_col].mean()
        control_mean = period_data[period_data[treatment_col] == 0][outcome_col].mean()
        diff = treated_mean - control_mean
        results.append({'time': t, 'diff': diff})

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['time'], results_df['diff'], marker='o')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero difference')
    plt.axvline(x=treatment_time, color='gray', linestyle='--', label='Treatment')
    plt.xlabel('Time')
    plt.ylabel('Treatment-Control Difference')
    plt.title('Parallel Trends Test')
    plt.legend()
    plt.tight_layout()

    return results_df
```

### Visual Inspection

Plot the outcome over time for treatment and control groups. Pre-treatment trends should be similar in level and slope, and not diverging before the treatment.

---

## Interpretation Workshop

### Reading DiD Studies

Key questions:
1. **What is the identifying assumption?** (Usually parallel trends)
2. **How is it tested?** (Pre-treatment trends, placebo tests)
3. **What is the comparison group?** (Is it credible?)
4. **Are there anticipation effects?** (Did units respond before the treatment?)
5. **Is the estimate ATT or ATE?** (DiD estimates ATT under parallel trends)

---

## Practical Application

### Standard DiD Implementation

```python
import statsmodels.api as sm
import numpy as np

def difference_in_differences(df, outcome, treatment, post, covariates=None):
    """Standard 2x2 DiD estimation."""

    df = df.copy()
    df['did'] = df[treatment] * df[post]

    if covariates:
        X = df[[treatment, post, 'did'] + covariates]
    else:
        X = df[[treatment, post, 'did']]

    X = sm.add_constant(X)
    model = sm.OLS(df[outcome], X).fit()

    did_coef = model.params['did']
    did_se = model.bse['did']

    print(f"DiD Estimate: {did_coef:.4f}")
    print(f"Std Error: {did_se:.4f}")

    return model
```

### Event Study Specification

```python
def event_study(df, outcome, treatment, time_col, treatment_time):
    """Event study with dynamic treatment effects."""

    df = df.copy()
    df['relative_time'] = df[time_col] - treatment_time

    # Omit the period just before treatment as reference
    formula = f"{outcome} ~ C(relative_time) * {treatment}"

    model = sm.OLS.from_formula(formula, data=df).fit()

    return model
```

---

## Limitations

- **Parallel trends untestable**: Only testable in pre-treatment periods
- **Confounding by co-interventions**: Other policies may change simultaneously
- **General equilibrium effects**: Control group may be affected by the treatment
- **Composition changes**: Units may enter or exit the sample over time

---

## Exercises

1. **Parallel trends test**: Using a dataset of your choice, implement the parallel trends test. What do you find?

2. **Event study**: Create an event study plot for a policy change. Are there pre-trends? Are there dynamic treatment effects?

3. **Sensitivity analysis**: Implement Oster (2019) sensitivity analysis for DiD. How much selection on unobservables would be needed to explain away the result?

4. **Critique**: Find a published DiD study. Evaluate the parallel trends assumption. What are the threats?

---

## Projects

### Project 1: Synthetic DiD
Implement synthetic difference-in-differences (Arkhangelsky et al., 2021) and compare to standard DiD.

### Project 2: Staggered DiD
Implement Callaway and Sant'Anna (2021) for staggered treatment adoption. Compare to two-way fixed effects.
