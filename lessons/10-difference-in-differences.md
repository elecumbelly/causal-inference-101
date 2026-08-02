---
title: "Lesson 10: Difference-in-Differences"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 10: Difference-in-Differences

## Opening Story: Minimum Wages and Employment

In 1992, New Jersey raised its minimum wage while neighbouring Pennsylvania did not. Card and Krueger compared how fast-food employment changed on the two sides of the state border before and after the policy.

Difference-in-differences (DiD) does not assume that the states have equal employment levels. It assumes that, without the policy, their *changes* would have followed parallel paths. Pennsylvania supplies an estimate of New Jersey’s missing post-policy counterfactual trend.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the parallel trends assumption
2. Implement two-way fixed effects regression
3. Conduct event studies
4. Test for pre-trends
5. Apply DiD to policy evaluation

---

```{figure} ../figures/instructional/did-trends.svg
:name: lesson-10-did-trends
:alt: DiD estimates the gap between the observed treated trajectory and its parallel-trends counterfactual.
:width: 100%

DiD estimates the gap between the observed treated trajectory and its parallel-trends counterfactual.
```

---

## 10.1 The DiD Framework

### The Setup

- Treatment group: New Jersey restaurants exposed to the wage increase
- Control group: Pennsylvania restaurants without the change
- Pre-period: Before the increase
- Post-period: After the increase

### The Parallel Trends Assumption

The key assumption is that without treatment, California's smoking trend would have been the same as other states:

$$E[Y_{it}(0) - Y_{it-1}(0) | D_i = 1] = E[Y_{it}(0) - Y_{it-1}(0) | D_i = 0]$$

### The DiD Estimator

$$\hat{\tau}_{DiD} = (Y_{post,treat} - Y_{pre,treat}) - (Y_{post,control} - Y_{pre,control})$$

---

## 10.2 Two-Way Fixed Effects

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import linearmodels

np.random.seed(42)
n_states = 40
n_years = 10

# Generate panel data
states = np.repeat(range(n_states), n_years)
years = np.tile(range(n_years), n_states)
treated = (states < 10).astype(int)  # First 10 states treated
post = (years >= 5).astype(int)  # Treatment starts at year 5

# Fixed effects
state_fe = np.random.normal(0, 1, n_states)[states]
time_fe = np.random.normal(0, 0.5, n_years)[years]

# Outcome with treatment effect
treatment_effect = 2.0
y = 10 + 0.5 * years + state_fe + time_fe + \
    treatment_effect * treated * post + np.random.normal(0, 1, n_states * n_years)

# Create DataFrame
df = pd.DataFrame({
    'state': states,
    'year': years,
    'treated': treated,
    'post': post,
    'y': y
})

# DiD regression
df['did'] = df['treated'] * df['post']
X = sm.add_constant(df[['treated', 'post', 'did']])
model = sm.OLS(df['y'], X).fit()
print("DiD estimate:", round(model.params['did'], 3))
print("True effect:", treatment_effect)
```

---

## 10.3 Event Study

```python
# Event study: leads and lags
for t in range(-5, 6):
    df[f'lead_lag_{t}'] = (df['treated'] == 1) & (df['year'] - 5 == t)

# Include leads (omitting the period before treatment as reference)
leads_lags = [f'lead_lag_{t}' for t in range(-5, 6) if t != -1]
X_event = sm.add_constant(df[leads_lags])
model_event = sm.OLS(df['y'], X_event).fit()

# Plot coefficients
import matplotlib.pyplot as plt

coef = [model_event.params[f'lead_lag_{t}'] for t in range(-5, 6) if t != -1]
se = [model_event.bse[f'lead_lag_{t}'] for t in range(-5, 6) if t != -1]

plt.figure(figsize=(10, 6))
plt.errorbar(range(-5, 6), coef, yerr=1.96 * np.array(se), fmt='o-', capsize=5)
plt.axvline(x=0, color='red', linestyle='--', label='Treatment')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
plt.xlabel('Time relative to treatment')
plt.ylabel('Coefficient')
plt.title('Event Study')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 10.4 Testing Pre-Trends

If pre-treatment coefficients are significantly different from zero, the parallel trends assumption may be violated.

---

## Modern DiD with Staggered Adoption

When groups adopt treatment at different times, a single two-way fixed-effects coefficient can compare newly treated groups with already-treated groups. With heterogeneous or dynamic effects, those comparisons may receive negative or unintuitive weights and need not estimate a meaningful average treatment effect.

A safer workflow is to:

1. Define effects by **group and adoption time**.
2. Use never-treated or not-yet-treated units as explicit controls.
3. Aggregate group-time effects only after examining their heterogeneity.
4. Plot event-time estimates with simultaneous uncertainty intervals.
5. Avoid treating a non-significant pre-trend test as proof of parallel trends.

Anticipation, compositional changes and treatment reversals require separate design choices; they are not fixed by adding more regression terms.

---

## 10.5 Common Mistakes

1. **Parallel trends violation**: Always plot pre-trends
2. **Anticipation effects**: Treatment may affect outcomes before implementation
3. **Spillovers**: Treatment in one unit affects others
4. **Time-varying confounders**: Factors that change differently across groups

---

## 10.6 Knowledge Check

### Multiple Choice

1. **The parallel trends assumption states:**
   - A) Trends are the same before and after treatment
   - B) Treatment and control groups have the same trends without treatment
   - C) There are no trends
   - D) Trends are parallel after treatment

2. **DiD estimates the causal effect when:**
   - A) Parallel trends hold
   - B) Pre-trends are zero
   - C) Both A and B
   - D) Neither A nor B

3. **An event study tests:**
   - A) Post-treatment effects
   - B) Pre-treatment trends
   - C) Both A and B
   - D) Neither A nor B

4. **Anticipation effects cause:**
   - A) Underestimation
   - B) Overestimation
   - C) No bias
   - D) Bias in unknown direction

5. **Spillovers violate:**
   - A) SUTVA
   - B) Parallel trends
   - C) Independence
   - D) Positivity

### Short Answer

6. **Explain why DiD is called a "natural experiment."**

7. **What happens if pre-trends are not parallel?**

8. **How can you test for anticipation effects?**

9. **Describe the role of the control group in DiD.**

10. **Give an example of a policy evaluation using DiD.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Difference-in-differences and natural experiments**](https://www.youtube.com/watch?v=tT8xLRS_cRQ&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=58)

Connects the parallel-trends assumption to counterfactual outcome trajectories.

**Active-viewing prompt:** Explain why similar pre-treatment levels are not the same as parallel trends.
```

---

## 10.7 Summary

1. **DiD** compares changes over time between treatment and control groups
2. **Parallel trends** is the key identifying assumption
3. **Event studies** test for pre-trends
4. **Two-way fixed effects** is the standard implementation
5. **Spillovers** and **anticipation effects** are threats to validity

---

## 10.8 Further Reading

- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Callaway, B. & Sant'Anna, P.H. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*.


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


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/10-difference-in-differences.ipynb)
- [Download the practice lab](../labs/lab10-difference-in-differences-practice.ipynb)
- [Download the lab solution](../solutions/lab10-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
