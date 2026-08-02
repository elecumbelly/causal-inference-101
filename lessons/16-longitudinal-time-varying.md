---
title: "Lesson 16: Longitudinal Data and Time-Varying Treatments"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 16: Longitudinal Data and Time-Varying Treatments

## Opening Story: HIV Treatment Timing

When should HIV patients start antiretroviral therapy? Starting too early might cause unnecessary side effects; starting too late might allow the virus to damage the immune system. The optimal timing depends on how treatment effects change over time and how patient characteristics evolve.

This is a time-varying treatment problem: the treatment decision at each time point depends on the patient's current and past history, and the treatment itself changes the patient's trajectory.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the challenges of time-varying treatments
2. Define marginal structural models
3. Implement inverse probability of treatment weighting (IPTW)
4. Conduct g-computation
5. Handle time-varying confounding

---

```{figure} ../figures/instructional/longitudinal-feedback.svg
:name: lesson-16-longitudinal-feedback
:alt: Treatment-confounder feedback is why ordinary adjustment fails in longitudinal settings.
:width: 100%

Treatment-confounder feedback is why ordinary adjustment fails in longitudinal settings.
```

---

## 16.1 Time-Varying Confounding

### The Problem

When treatment affects future covariates, and those covariates affect future treatment and outcomes, we have time-varying confounding. Standard methods like regression adjustment or matching fail because:

1. Conditioning on post-treatment variables introduces bias
2. Not conditioning leaves confounding unaddressed

### Example

```
Treatment → Blood Pressure → Future Treatment
    ↓                              ↓
Outcome ←←←←←←←←←←←←←←←←←←←←←←←←
```

Blood pressure is affected by treatment and affects future treatment decisions. Conditioning on blood pressure blocks part of the treatment effect.

---

## 16.2 Marginal Structural Models

### The Framework

Model the marginal (population-averaged) potential outcomes:

$$E[Y^{\bar{a}}] = g(\bar{a}; \psi)$$

where $\bar{a}$ is a treatment history and $\psi$ are parameters.

### IPTW Estimation

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
n_time = 5

# Generate longitudinal data
data = []
for i in range(n):
    # Initial covariates
    L0 = np.random.normal(0, 1)

    for t in range(n_time):
        # Treatment at time t
        prob_treat = 1 / (1 + np.exp(-(-0.5 + 0.5 * L0)))
        A_t = np.random.binomial(1, prob_treat)

        # Outcome at time t
        Y_t = 0.5 * L0 + 1.0 * A_t + np.random.normal(0, 0.5)

        data.append({
            'id': i,
            'time': t,
            'L': L0,
            'A': A_t,
            'Y': Y_t
        })

        # Update covariates
        L0 = L0 + 0.3 * A_t + np.random.normal(0, 0.2)

df = pd.DataFrame(data)

# IPTW weights
def calculate_iptw_weights(df):
    weights = np.ones(df['id'].nunique())

    for t in range(n_time):
        t_data = df[df['time'] == t]

        # Model treatment given history
        from sklearn.linear_model import LogisticRegression

        if t == 0:
            X = t_data[['L']].values
        else:
            # Include past treatment and covariates
            past_data = df[df['time'] < t].groupby('id').last().reset_index()
            X = t_data[['L']].merge(past_data[['id', 'A']], on='id')[['L', 'A']].values

        A = t_data['A'].values

        # Fit treatment model
        model = LogisticRegression()
        model.fit(X, A)
        prob = model.predict_proba(X)[:, 1]

        # Update weights
        w = np.where(A == 1, 1/prob, 1/(1-prob))
        unique_ids = t_data['id'].values
        weights[unique_ids] *= w

    return weights

# Calculate stabilized weights
def calculate_stabilized_weights(df):
    weights = np.ones(df['id'].nunique())

    for t in range(n_time):
        t_data = df[df['time'] == t]

        # Numerator: marginal probability of treatment
        prob_marginal = t_data['A'].mean()

        # Denominator: conditional probability
        from sklearn.linear_model import LogisticRegression

        if t == 0:
            X = t_data[['L']].values
        else:
            past_data = df[df['time'] < t].groupby('id').last().reset_index()
            X = t_data[['L']].merge(past_data[['id', 'A']], on='id')[['L', 'A']].values

        A = t_data['A'].values
        model = LogisticRegression()
        model.fit(X, A)
        prob_conditional = model.predict_proba(X)[:, 1]

        # Stabilized weight
        w = np.where(A == 1, prob_marginal/prob_conditional,
                     (1-prob_marginal)/(1-prob_conditional))

        unique_ids = t_data['id'].values
        weights[unique_ids] *= w

    return weights
```

---

## 16.3 G-computation

An alternative to IPTW that models the conditional expectation of outcomes.

---

## 16.4 Common Mistakes

1. **Conditioning on post-treatment variables**: Use IPTW or g-computation
2. **Ignoring time-varying confounding**: Standard methods fail
3. **Model misspecification**: Validate treatment and outcome models
4. **Extreme weights**: Use trimming or stabilized weights

---

## 16.5 Knowledge Check

### Multiple Choice

1. **Time-varying confounding occurs when:**
   A) Treatment affects future covariates
   B) Covariates affect future treatment
   C) Both A and B
   D) Neither

2. **IPTW handles time-varying confounding by:**
   A) Conditioning on covariates
   B) Weighting by inverse probability of treatment
   C) Randomizing treatment
   D) Ignoring confounding

3. **Stabilized weights:**
   A) Reduce variance
   B) Increase bias
   C) Have no effect
   D) Are always necessary

4. **G-computation:**
   A) Is always better than IPTW
   B) Requires correct outcome model
   C) Doesn't need any models
   D) Is only for experiments

5. **Post-treatment variables:**
   A) Should always be controlled
   B) Should never be controlled
   C) Should be controlled only sometimes
   D) Don't exist

### Short Answer

6. **Explain why standard regression fails with time-varying confounding.**

7. **How do IPTW weights account for time-varying confounding?**

8. **What are the advantages of stabilized weights?**

9. **When might g-computation be preferred over IPTW?**

10. **Give an example of a time-varying treatment problem.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Brady Neal’s causal inference lecture series**](https://www.youtube.com/playlist?list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0)

Use the identification, estimation, and sensitivity lectures as prerequisites for g-methods, treatment-confounder feedback, and longitudinal interventions.

**Active-viewing prompt:** Draw the time ordering before deciding whether standard regression is valid.
```

---

## 16.6 Summary

1. **Time-varying confounding** requires special methods
2. **IPTW** weights observations by treatment probability
3. **Stabilized weights** reduce variance
4. **G-computation** models the outcome directly
5. **Sequential ignorability** is the key assumption

---

## 16.7 Further Reading

- Robins, J.M., Hernán, M.A., & Brumback, B. (2000). "Marginal Structural Models and Causal Inference in Epidemiology." *Epidemiology*.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.


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


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/16-longitudinal-time-varying.ipynb)
- [Download the practice lab](../labs/lab16-longitudinal-time-varying-practice.ipynb)
- [Download the lab solution](../solutions/lab16-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
