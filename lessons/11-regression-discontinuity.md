---
title: "Lesson 11: Regression Discontinuity Design"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 11: Regression Discontinuity Design

## Opening Story: Scholarship Eligibility

Suppose students scoring 80 or above on an entrance examination receive a scholarship while those scoring below 80 do not. A student scoring 79.9 and one scoring 80.1 are likely similar in background and preparation, yet the rule assigns them different treatment status.

This is the essence of regression discontinuity design (RDD): if potential outcomes evolve smoothly through a cutoff and people cannot precisely manipulate their score, a discontinuity in outcomes identifies a **local** causal effect at that threshold.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain the RDD framework
2. Implement sharp and fuzzy RDD
3. Conduct bandwidth selection
4. Test for manipulation of the running variable
5. Interpret local average treatment effects

---

```{figure} ../figures/instructional/rdd-cutoff.svg
:name: lesson-11-rdd-cutoff
:alt: RDD identifies a local jump at the treatment threshold.
:width: 100%

RDD identifies a local jump at the treatment threshold.
```

---

## 11.1 The RDD Framework

### Sharp RDD

Treatment is deterministically assigned based on a cutoff:

$$T_i = \mathbb{1}(X_i \geq c)$$

The causal effect at the cutoff:

$$\tau = \lim_{x \downarrow c} E[Y_i | X_i = x] - \lim_{x \uparrow c} E[Y_i | X_i = x]$$

### Fuzzy RDD

Treatment is not perfectly determined by the cutoff, but there's a discontinuity in the probability of treatment:

$$P(T_i = 1 | X_i = x)$$

has a jump at $x = c$.

---

## 11.2 Sharp RDD Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 1000

# Running variable (e.g., test score)
X = np.random.uniform(0, 100, n)

# Treatment assignment (sharp cutoff at 50)
cutoff = 50
T = (X >= cutoff).astype(int)

# Outcome (effect of 2 for those above cutoff)
Y = 10 + 0.5 * X + 2 * T + np.random.normal(0, 2, n)

# Fit separate regressions on each side
left_mask = X < cutoff
right_mask = X >= cutoff

model_left = LinearRegression()
model_left.fit(X[left_mask].reshape(-1, 1), Y[left_mask])

model_right = LinearRegression()
model_right.fit(X[right_mask].reshape(-1, 1), Y[right_mask])

# Estimate treatment effect at cutoff
effect_left = model_left.predict([[cutoff]])[0]
effect_right = model_right.predict([[cutoff]])[0]
effect_rdd = effect_right - effect_left

print(f"Effect at cutoff: {effect_rdd:.3f}")
print(f"True effect: 2.0")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X[left_mask], Y[left_mask], alpha=0.3, s=10, label='Control')
plt.scatter(X[right_mask], Y[right_mask], alpha=0.3, s=10, label='Treated')

x_range = np.linspace(0, 100, 100)
plt.plot(x_range[x_range < cutoff], model_left.predict(x_range[x_range < cutoff].reshape(-1, 1)),
         color='blue', linewidth=2, label='Control fit')
plt.plot(x_range[x_range >= cutoff], model_right.predict(x_range[x_range >= cutoff].reshape(-1, 1)),
         color='red', linewidth=2, label='Treated fit')

plt.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')
plt.xlabel('Running Variable')
plt.ylabel('Outcome')
plt.title('Sharp Regression Discontinuity Design')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 11.3 Fuzzy RDD

```python
# Fuzzy RDD: treatment probability jumps at cutoff
prob_treat = 0.2 + 0.6 * (X >= cutoff).astype(int)
T_fuzzy = np.random.binomial(1, prob_treat)

# Reduced form
Y_fuzzy = 10 + 0.5 * X + 2 * T_fuzzy + np.random.normal(0, 2, n)

# Wald estimator for fuzzy RDD
# Effect = (jump in outcome) / (jump in treatment probability)
# This is similar to IV with the cutoff as the instrument
```

---

## 11.4 Bandwidth Selection

The choice of bandwidth involves a bias-variance tradeoff:
- **Narrow bandwidth**: Less bias, more variance
- **Wide bandwidth**: More bias, less variance

---

## Robust Estimation and Inference

RDD is a boundary-estimation problem. High-order global polynomials can behave erratically near the cutoff and should not be the default. Modern practice fits low-order local polynomials within a data-driven bandwidth and reports robust bias-corrected confidence intervals.

A credible analysis reports:

- the running-variable density near the threshold;
- covariate continuity and predetermined outcomes;
- estimates across reasonable bandwidths and polynomial orders;
- the number of observations on each side;
- conventional and robust bias-corrected uncertainty;
- the institutional rule establishing that no other policy changes at the same cutoff.

The estimand remains local: evidence at the threshold does not by itself identify effects for units far away.

---

## 11.5 Common Mistakes

1. **Manipulation of running variable**: Test for bunching at the cutoff
2. **Wrong bandwidth**: Use optimal bandwidth selection methods
3. **Extrapolating beyond the cutoff**: RDD estimates effects only at the cutoff
4. **Ignoring covariates**: Use covariates to improve precision

---

## 11.6 Knowledge Check

### Multiple Choice

1. **RDD estimates the effect:**
   - A) At the cutoff only
   - B) For everyone
   - C) For units near the cutoff
   - D) Both A and C

2. **Sharp RDD assumes:**
   - A) Perfect compliance
   - B) No compliance
   - C) Partial compliance
   - D) Random compliance

3. **The running variable is:**
   - A) The treatment
   - B) The outcome
   - C) The variable determining treatment
   - D) A confounder

4. **Manipulation of the running variable:**
   - A) Is good
   - B) Violates the design
   - C) Has no effect
   - D) Is necessary

5. **Fuzzy RDD is similar to:**
   - A) OLS
   - B) IV
   - C) Matching
   - D) DiD

### Short Answer

6. **Explain why RDD gives causal estimates.**

7. **What is the difference between sharp and fuzzy RDD?**

8. **How can you test for manipulation of the running variable?**

9. **Why is the LATE interpretation important in fuzzy RDD?**

10. **Give an example of a policy that uses a cutoff for eligibility.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Regression discontinuity in the natural-experiments toolkit**](https://www.youtube.com/watch?v=tT8xLRS_cRQ&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=58)

Situates discontinuity designs alongside IV and panel-data strategies.

**Active-viewing prompt:** Name the estimand at the cutoff and why it need not generalize far from it.
```

---

## 11.7 Summary

1. **RDD** exploits cutoffs in treatment assignment
2. **Sharp RDD** has perfect compliance at the cutoff
3. **Fuzzy RDD** has imperfect compliance
4. **Bandwidth selection** is crucial for validity
5. **LATE** is the parameter of interest

---

## 11.8 Further Reading

- Lee, D.S. & Lemieux, T. (2010). "Regression Discontinuity Designs in Economics." *Journal of Economic Literature*.
- Cattaneo, M.D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*. Cambridge University Press.


---

## Worked Examples

### Example 1: Economics — Class Size Threshold

In Israeli schools, classes with more than 40 students are split. This creates a discontinuity in class size at the threshold. Comparing students just above and below 40 estimates the effect of class size on achievement.

### Example 2: Health — Kidney Transplant Priority

Patients with GFR < 20 are prioritized for kidney transplants. Comparing patients just above and below 20 estimates the effect of transplant priority on survival.

### Example 3: Policy — Scholarship Eligibility

Students with SAT scores above 1200 receive a scholarship. Comparing students just above and below 1200 estimates the effect of financial support on college completion.

### Example 4: Technology — Feature Rollout

A platform releases a new feature to users with > 1000 followers. Comparing users just above and below 1000 estimates the feature's effect on engagement.

---

## Diagnostics: RDD Validity

### McCrary Density Test

```python
import numpy as np
import matplotlib.pyplot as plt

def mccrary_density_test(running_var, cutoff, bandwidth=None):
    """Test for manipulation of the running variable."""

    if bandwidth is None:
        bandwidth = np.std(running_var) * 0.2

    below = running_var[running_var < cutoff]
    above = running_var[running_var >= cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(running_var, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=cutoff, color='r', linestyle='--', label='Cutoff')
    axes[0].set_xlabel('Running Variable')
    axes[0].set_title('Distribution of Running Variable')
    axes[0].legend()

    # Density estimate
    from sklearn.neighbors import KernelDensity

    x_grid = np.linspace(cutoff - 3*bandwidth, cutoff + 3*bandwidth, 200).reshape(-1, 1)

    kde_below = KernelDensity(bandwidth=bandwidth).fit(below.reshape(-1, 1))
    kde_above = KernelDensity(bandwidth=bandwidth).fit(above.reshape(-1, 1))

    log_density_below = kde_below.score_samples(x_grid)
    log_density_above = kde_above.score_samples(x_grid)

    axes[1].plot(x_grid, np.exp(log_density_below), label='Below cutoff', linewidth=2)
    axes[1].plot(x_grid, np.exp(log_density_above), label='Above cutoff', linewidth=2)
    axes[1].axvline(x=cutoff, color='r', linestyle='--', label='Cutoff')
    axes[1].set_xlabel('Running Variable')
    axes[1].set_title('Density at Cutoff')
    axes[1].legend()

    plt.tight_layout()

    density_below = np.exp(kde_below.score_samples([[cutoff]]))[0]
    density_above = np.exp(kde_above.score_samples([[cutoff]]))[0]
    ratio = density_above / density_below

    print(f"Density ratio at cutoff: {ratio:.3f}")
    print("Values close to 1 suggest no manipulation")

    return fig, ratio
```

### Covariate Balance at the Cutoff

```python
def covariate_balance_rdd(df, running_var, cutoff, covariates, bandwidth=None):
    """Check covariate balance at the discontinuity."""

    if bandwidth is None:
        bandwidth = np.std(df[running_var]) * 0.2

    near_cutoff = df[(df[running_var] >= cutoff - bandwidth) &
                     (df[running_var] <= cutoff + bandwidth)]

    below = near_cutoff[near_cutoff[running_var] < cutoff]
    above = near_cutoff[near_cutoff[running_var] >= cutoff]

    print("Covariate Balance at Discontinuity:")
    for cov in covariates:
        diff = above[cov].mean() - below[cov].mean()
        pooled_std = np.sqrt((above[cov].var() + below[cov].var()) / 2)
        smd = abs(diff) / pooled_std if pooled_std > 0 else 0
        status = "OK" if smd < 0.1 else "WARN"
        print(f"  {cov}: SMD = {smd:.3f} [{status}]")
```

---

## Interpretation Workshop

### Reading RDD Studies

Key questions:
1. **What is the running variable and cutoff?** Is the threshold clearly defined?
2. **Is there manipulation?** (McCrary test, covariate balance)
3. **What bandwidth was used?** Narrower = more local, wider = more power
4. **What polynomial order?** Higher order can introduce bias
5. **Is the result robust** to bandwidth and polynomial choices?

### Local vs Global Estimates

RDD estimates the effect **at the cutoff** — a local average treatment effect. This may not generalize to units far from the threshold.

---

## Practical Application

### Standard RDD Estimation

```python
import numpy as np
import statsmodels.api as sm

def rdd_estimate(df, running_var, outcome, cutoff, bandwidth=None, polynomial=1):
    """Estimate RDD effect using local polynomial regression."""

    df = df.copy()
    df['centered'] = df[running_var] - cutoff

    if bandwidth is not None:
        df = df[df['centered'].abs() <= bandwidth]

    df['treated'] = (df['centered'] >= 0).astype(int)

    if polynomial == 1:
        X = df[['centered', 'treated']]
    elif polynomial == 2:
        df['centered_sq'] = df['centered'] ** 2
        X = df[['centered', 'centered_sq', 'treated']]
    else:
        for p in range(1, polynomial + 1):
            df[f'centered_{p}'] = df['centered'] ** p
        X = df[[f'centered_{p}' for p in range(1, polynomial + 1)] + ['treated']]

    X = sm.add_constant(X)
    model = sm.OLS(df[outcome], X).fit()

    effect = model.params['treated']
    se = model.bse['treated']

    print(f"RDD Estimate: {effect:.4f} (SE: {se:.4f})")

    return model
```

### Robustness Checks

1. **Vary bandwidth**: Use triangular kernel, vary bandwidth from 0.5h to 2h
2. **Vary polynomial order**: Compare linear, quadratic, cubic
3. **Placebo cutoffs**: Test at cutoffs where no effect is expected
4. **Covariate adjustment**: Add covariates to the regression

---

## Limitations

- **Local estimate**: Only valid at the cutoff
- **Sensitivity to bandwidth**: Results can vary with bandwidth choice
- **Manipulation**: If units can manipulate the running variable, the estimate is biased
- **Density continuity**: Requires no bunching at the cutoff

---

## Exercises

1. **Implement RDD**: Use the provided code to estimate an RDD effect. Vary the bandwidth and polynomial order. How stable are the results?

2. **McCrary test**: Implement the McCrary density test. Is there evidence of manipulation?

3. **Placebo test**: Choose a placebo cutoff where no effect should exist. Does the RDD estimate equal zero?

4. **Fuzzy RDD**: Modify the code to handle fuzzy discontinuities (where treatment probability changes but does not jump from 0 to 1).

---

## Projects

### Project 1: RDD Robustness
Conduct a comprehensive RDD analysis with multiple bandwidth choices, polynomial orders, kernel weighting, covariate adjustment, and placebo tests.

### Project 2: Fuzzy RDD
Implement fuzzy RDD using 2SLS. Compare to sharp RDD estimates.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/11-regression-discontinuity.ipynb)
- [Download the practice lab](../labs/lab11-regression-discontinuity-practice.ipynb)
- [Download the lab solution](../solutions/lab11-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
