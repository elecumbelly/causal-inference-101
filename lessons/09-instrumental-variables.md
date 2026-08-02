---
title: "Lesson 9: Instrumental Variables"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 9: Instrumental Variables

## Opening Story: Returns to Education

How much does education increase earnings? This is one of the most studied questions in economics. The problem is that education is correlated with ability, family background, and motivation—all of which affect earnings.

In 1993, David Card used geographic proximity to a four-year college as an instrument for schooling. Proximity predicted educational attainment, but the exclusion restriction—that proximity affects earnings only through schooling—is a substantive assumption, not something the data can prove. Local labour markets, family location choices or migration could threaten it.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define instrumental variables and the exclusion restriction
2. Verify instrument validity conditions
3. Implement two-stage least squares (2SLS)
4. Test for weak instruments
5. Apply IV methods to real-world problems

---

```{figure} ../figures/instructional/iv-graph.svg
:name: lesson-09-iv-graph
:alt: An instrument affects the outcome through treatment, not through a direct path.
:width: 100%

An instrument affects the outcome through treatment, not through a direct path.
```

---

## 9.1 The IV Framework

### The Problem

We want to estimate the causal effect of $X$ on $Y$, but there's an unobserved confounder $U$:

$$X = \alpha_1 + \beta_1 Z + \gamma_1 U + \epsilon_1$$
$$Y = \alpha_2 + \beta_2 X + \gamma_2 U + \epsilon_2$$

### The Solution

Find an instrument $Z$ that:
1. **Relevance**: $Z$ is correlated with $X$
2. **Exclusion restriction**: $Z$ affects $Y$ only through $X$
3. **Independence**: $Z$ is independent of $U$

For a local average treatment effect interpretation, we additionally require **monotonicity**: changing the instrument in the encouraging direction does not make anyone move away from treatment.

---

## 9.2 Two-Stage Least Squares (2SLS)

### Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

np.random.seed(42)
n = 1000

# Generate data
U = np.random.normal(0, 1, n)  # Unobserved confounder
Z = np.random.normal(0, 1, n)  # Instrument
X = 0.5 * Z + 0.8 * U + np.random.normal(0, 0.5, n)  # Endogenous variable
Y = 2 * X + 0.8 * U + np.random.normal(0, 1, n)  # Outcome

# OLS (biased)
ols_model = sm.OLS(Y, sm.add_constant(X)).fit()
print("OLS estimate (biased):", round(ols_model.params[1], 3))

# IV/2SLS
iv_model = IV2SLS(Y, sm.add_constant(X), Z).fit()
print("IV estimate (consistent):", round(iv_model.params.iloc[0], 3))
print("True effect: 2.0")
```

---

## 9.3 Instrument Validity

### Relevance

Test with first-stage F-statistic:

```python
first_stage = sm.OLS(X, sm.add_constant(Z)).fit()
f_stat = first_stage.fvalue
print(f"First-stage F-statistic: {f_stat:.1f}")
print(f"Rule of thumb: F > 10 for strong instrument")
```

### Exclusion Restriction

Cannot be tested empirically—must argue theoretically that the instrument affects the outcome only through treatment.

### Independence

Requires that the instrument is as good as randomly assigned. Often argued through institutional knowledge.

---

## 9.4 Common Mistakes

1. **Weak instruments**: Test F-statistic > 10
2. **Invalid instruments**: The exclusion restriction is untestable
3. **Local average treatment effects**: IV estimates effects for compliers only

---

## 9.5 Discussion Questions

1. **Card's Study**: Why might proximity to college be a valid instrument?

2. **Exclusion Restriction**: How can you argue that an instrument satisfies the exclusion restriction?

3. **LATE**: What is the local average treatment effect, and why does it matter?

4. **Weak Instruments**: What happens if your instrument is weak?

5. **Alternative Instruments**: What are some other instruments used in economics?

---

## 9.6 Knowledge Check

### Multiple Choice

1. **An instrument must satisfy:**
   - A) Relevance and exclusion restriction
   - B) Relevance only
   - C) Exclusion restriction only
   - D) Neither

2. **The first-stage F-statistic tests:**
   - A) Exclusion restriction
   - B) Relevance
   - C) Independence
   - D) Exogeneity

3. **2SLS estimates the:**
   - A) Average treatment effect
   - B) Local average treatment effect
   - C) Conditional average treatment effect
   - D) Total treatment effect

4. **Weak instruments cause:**
   - A) Bias
   - B) Inconsistency
   - C) Inefficiency
   - D) All of the above

5. **The exclusion restriction is:**
   - A) Testable
   - B) Untestable
   - C) Always satisfied
   - D) Irrelevant

### Short Answer

6. **Explain the local average treatment effect.**

7. **Why is the exclusion restriction untestable?**

8. **Describe the consequences of using an invalid instrument.**

9. **How can you strengthen a weak instrument?**

10. **Give an example of a natural experiment that could serve as an instrument.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Instrumental variables**](https://www.youtube.com/watch?v=Mco16tUSA-U&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=53)

Builds IV from relevance, exclusion, and monotonicity rather than from the 2SLS recipe alone.

**Active-viewing prompt:** State the population for whom the local average treatment effect applies.
```

---

## 9.7 Summary

1. **IV methods** address unmeasured confounding
2. **2SLS** is the standard estimation method
3. **Relevance** is testable; **exclusion restriction** is not
4. **LATE** is the parameter of interest
5. **Weak instruments** are a major concern

---

## 9.8 Further Reading

- Angrist, J.D. & Krueger, A.B. (2001). "Instrumental Variables and the Search for Identification." *Journal of Economic Perspectives*.
- Card, D. (1999). "The Causal Effect of Education on Earnings." *Handbook of Labor Economics*.


---

## Worked Examples

### Example 1: Economics — Returns to Education (Card 1991)

**Instrument**: Distance to nearest college
**Treatment**: Years of education
**Outcome**: Log earnings

**Assumptions:**
1. **Relevance**: People closer to colleges attend more education (check: F-statistic > 10)
2. **Exclusion restriction**: Proximity affects earnings only through education
3. **Independence**: Proximity is independent of unobserved ability

**Potential violations**: People near colleges may have different labor market access, cost of living, or social networks that directly affect earnings.

### Example 2: Health — Effect of Smoking on Birth Weight

**Instrument**: Cigarette taxes in the mother's state of residence
**Treatment**: Cigarettes smoked per day during pregnancy
**Outcome**: Birth weight

**Logic**: Higher taxes reduce smoking, affecting birth weight only through smoking.

**Assumptions to defend**: Taxes do not affect birth weight through other channels (e.g., maternal stress from financial burden).

### Example 3: Technology — Effect of Internet Access on Employment

**Instrument**: Distance to the nearest broadband hub
**Treatment**: Broadband internet access
**Outcome**: Employment probability

**Challenge**: Distance to broadband hubs may correlate with urbanization, which directly affects employment.

### Example 4: Policy — Effect of Prison on Recidivism

**Instrument**: Randomly assigned judge harshness
**Treatment**: Prison sentence (vs probation)
**Outcome**: Recidivism within 3 years

**Logic**: Judges have different sentencing tendencies, creating quasi-random variation in sentence severity.

---

## Diagnostics: Instrument Strength

### The First-Stage F-Statistic

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def first_stage_f(Z, X, covariates=None):
    """Calculate first-stage F-statistic for IV regression."""
    if covariates is not None:
        X_first = np.column_stack([Z, covariates])
    else:
        X_first = Z.reshape(-1, 1)

    first_stage = LinearRegression().fit(X_first, X)

    ss_res = np.sum((X - first_stage.predict(X_first)) ** 2)
    ss_tot = np.sum((X - np.mean(X)) ** 2)
    r_sq = 1 - ss_res / ss_tot

    n = len(X)
    k = X_first.shape[1]
    f_stat = (r_sq / 1) / ((1 - r_sq) / (n - k - 1))

    print(f"First-stage F-statistic: {f_stat:.1f}")
    print(f"Rule of thumb: F > 10 indicates strong instrument")

    if f_stat < 10:
        print("WARNING: Weak instrument — IV estimates may be biased toward OLS")

    return f_stat
```

### Weak Instrument Consequences

- **Bias toward OLS**: IV estimates are biased toward the confounded OLS estimate
- **Invalid confidence intervals**: Standard errors are too small
- **Inference failure**: Rejection rates exceed nominal levels

---

## Interpretation Workshop

### Reading IV Studies

Key questions:
1. **What is the instrument?** Is it intuitive and plausible?
2. **Is the exclusion restriction defensible?** What are the threats?
3. **Is the instrument strong enough?** Check F-statistic > 10
4. **What population does the LATE apply to?** (Compliers only)

### LATE vs ATE

The IV estimand is the **Local Average Treatment Effect (LATE)**: the effect for *compliers* — units whose treatment status is changed by the instrument.

- **ATE**: Average effect for everyone
- **LATE**: Average effect for compliers
- These differ when treatment effects are heterogeneous

```python
def explain_late():
    print("In the returns-to-education example:")
    print("- Compliers: People who attend college BECAUSE they live near one")
    print("- Always-takers: People who attend college regardless of distance")
    print("- Never-takers: People who do not attend college regardless of distance")
    print("")
    print("The IV estimate tells us: What is the return to education FOR COMPLIERS?")
    print("This may differ from the ATE because compliers may have different")
    print("ability levels, motivations, or treatment response.")
```

---

## Practical Application

### Two-Stage Least Squares Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def two_stage_least_squares(Y, X, Z, covariates=None):
    """Implement 2SLS estimation."""
    n = len(Y)

    # First stage: Regress X on Z (and covariates)
    if covariates is not None:
        first_stage_X = np.column_stack([Z, covariates])
    else:
        first_stage_X = Z.reshape(-1, 1)

    first_stage = LinearRegression().fit(first_stage_X, X)
    X_hat = first_stage.predict(first_stage_X)

    # Second stage: Regress Y on X_hat (and covariates)
    if covariates is not None:
        second_stage_X = np.column_stack([X_hat, covariates])
    else:
        second_stage_X = X_hat.reshape(-1, 1)

    second_stage = LinearRegression().fit(second_stage_X, Y)

    print(f"IV Estimate: {second_stage.coef_[0]:.4f}")

    return second_stage.coef_[0]
```

---

## Limitations

- **LATE only**: Effect estimated is for compliers, not the general population
- **Exclusion restriction untestable**: Must be argued on theoretical grounds
- **Weak instruments**: Common problem, leads to biased estimates
- **Monotonicity assumption**: No defiers (people who do the opposite of what the instrument pushes them to do)

---

## Exercises

1. **Instrument evaluation**: For each scenario, evaluate whether the proposed instrument is valid: (a) Weather as an instrument for agricultural output, (b) Blood type as an instrument for medical treatment, (c) Season of birth as an instrument for education.

2. **Weak instrument simulation**: Generate data with a weak instrument (correlation of 0.1 with treatment). Show how IV estimates are biased toward OLS.

3. **LATE calculation**: In a dataset with known complier status, calculate the LATE and compare it to the ATE. When do they differ most?

4. **Exclusion restriction**: Write a defense of the exclusion restriction for using college proximity as an instrument for education. What are the strongest threats?

---

## Projects

### Project 1: IV Sensitivity Analysis
Implement sensitivity analysis for IV studies by varying the exclusion restriction assumption and showing how conclusions change.

### Project 2: Compare IV Methods
Compare 2SLS, LIML, and JIVE estimators on bias under weak instruments, coverage of confidence intervals, and finite sample performance.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/09-instrumental-variables.ipynb)
- [Download the practice lab](../labs/lab09-instrumental-variables-practice.ipynb)
- [Download the lab solution](../solutions/lab09-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
