

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
