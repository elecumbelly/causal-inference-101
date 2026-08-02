---
title: "Lesson 7: Regression for Causal Adjustment"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 7: Regression for Causal Adjustment

## Opening Story: Education and Earnings

Does education cause higher earnings? The answer seems obvious—college graduates earn more than high school graduates. But correlation isn't causation. People who attend college might be smarter, more motivated, or come from wealthier families. These factors affect both education and earnings.

The question is: can regression analysis help us estimate the causal effect of education on earnings? The answer is: yes, under certain conditions—but those conditions are often violated in practice.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Use regression as a tool for causal adjustment
2. Understand conditional expectations and the law of iterated expectations
3. Implement multiple regression for causal inference
4. Recognize when regression fails
5. Interpret regression coefficients causally under appropriate assumptions

---

## 7.1 Regression as Causal Adjustment

### The Core Idea

Under the **unconfoundedness assumption** (all confounders are observed and controlled), regression can estimate causal effects:

$$Y = \alpha + \tau T + \beta X + \epsilon$$

If treatment effects are constant and the conditional mean is correctly specified:
- $\tau$ is the common treatment effect and therefore the ATE
- $\beta$ captures the effect of confounders
- $\epsilon$ is independent of $T$ and $X$

With heterogeneous effects, the coefficient on $T$ is generally a model-dependent weighted average, not automatically the population ATE. Regression also cannot repair non-overlap or unmeasured confounding.

### When This Works

1. All confounders are measured and included in $X$
2. The functional form is correct (linear, additive)
3. No measurement error in confounders
4. No collinearity between $T$ and $X$
5. Positivity: each relevant covariate profile has a non-zero chance of receiving either treatment

---

## 7.2 Conditional Expectations

### The Law of Iterated Expectations

$$E[Y | X] = E[E[Y | T, X] | X]$$

This means: the expected outcome given covariates is the average of the conditional treatment effects, weighted by the probability of treatment.

### The Regression Interpretation

In a correctly specified regression:
$$E[Y | T, X] = \alpha + \tau T + \beta X$$

The coefficient $\tau$ is the average treatment effect, conditional on $X$.

---

## 7.3 Multiple Regression

### The Workhorse

Multiple regression is the most common method for causal adjustment:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

np.random.seed(42)
n = 1000

# Generate data with confounding
ability = np.random.normal(0, 1, n)
education = 12 + 2 * ability + np.random.normal(0, 1, n)
earnings = 20000 + 5000 * education + 10000 * ability + np.random.normal(0, 5000, n)

# Omitted variable bias
reg_omit = sm.OLS(earnings, sm.add_constant(education)).fit()
print("Effect of education (omitting ability):")
print(f"  Coefficient: ${reg_omit.params[1]:.2f}")
print(f"  True effect: $5000")

# Complete regression
X_complete = np.column_stack([education, ability])
reg_complete = sm.OLS(earnings, sm.add_constant(X_complete)).fit()
print("\nEffect of education (controlling for ability):")
print(f"  Coefficient: ${reg_complete.params[1]:.2f}")
```

---

## 7.4 When Regression Fails

### 1. Omitted Variable Bias

If important confounders are missing, regression estimates are biased.

### 2. Functional Form Misspecification

If the true relationship is nonlinear but we fit a linear model, estimates can be biased.

### 3. Measurement Error

Errors in measuring confounders can cause attenuation bias (coefficients biased toward zero).

### 4. Perfect Collinearity

If treatment is perfectly predicted by covariates, we can't estimate its effect.

---

## 7.5 Case Study: Returns to Education

### The Question

What is the causal effect of an additional year of education on earnings?

### The Challenge

- Ability affects both education and earnings
- Family background affects both
- Motivation affects both

### The Solution

Use instrumental variables (covered in Lesson 9) or carefully control for observed confounders.

### Typical Findings

- Naive regression: ~10% return per year
- After controlling for ability: ~7-8% return
- IV estimates: ~5-15% return (varies by study)

---

## 7.6 Python Workshop: Regression Diagnostics

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

# Simulate data
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
T = (0.5 * X1 + 0.3 * X2 + np.random.normal(0, 0.5, n)) > 0
T = T.astype(int)
Y = 2 * T + 3 * X1 + 1 * X2 + np.random.normal(0, 1, n)

# Fit regression
X = sm.add_constant(np.column_stack([T, X1, X2]))
model = sm.OLS(Y, X).fit()

print(model.summary())

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Residuals vs fitted
axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.5, s=10)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
from scipy import stats
stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Scale-Location
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.5, s=10)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals vs leverage
from statsmodels.graphics.regressionplots import plot_leverage_resid2
plot_leverage_resid2(model, ax=axes[1, 1])
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('../figures/07-regression-diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Measurement Error and Missing Data

Regression adjustment assumes the variables in the model mean what we think they mean and are observed under a defensible process.

- **Error in treatment** can mix treatment versions and blur the estimand.
- **Error in confounders** leaves residual confounding even after adjustment.
- **Error in outcomes** can bias effects when it differs by treatment status.
- **Complete-case analysis** is valid only under restrictive missingness conditions and can create selection bias.

Before modelling, draw a causal diagram that includes measurement and missingness indicators. Ask whether observation depends on treatment, the outcome, or their causes. Multiple imputation addresses missing data only under its assumptions; it does not recreate variables that were never measured or repair non-identifiability.

---

## 7.7 Common Mistakes

1. **Assuming linearity**: Always check functional form
2. **Including post-treatment variables**: Don't control for mediators
3. **Ignoring heteroskedasticity**: Use robust standard errors
4. **Over-controlling**: Control for confounders, not colliders or mediators

---

## 7.8 Discussion Questions

1. **Education Returns**: Why might naive regression overestimate the return to education?

2. **Functional Form**: What happens if you include a quadratic term for age when the true relationship is linear?

3. **Measurement Error**: How does measurement error in a confounder affect your estimate?

4. **Multiple Regression**: When might adding more variables increase bias?

5. **Regression vs. Matching**: When might matching be preferable to regression?

---

## 7.9 Knowledge Check

### Multiple Choice

1. **Regression estimates the causal effect when:**
   - A) All confounders are controlled
   - B) The functional form is correct
   - C) Both A and B
   - D) Neither A nor B

2. **Omitted variable bias occurs when:**
   - A) We include too many variables
   - B) We exclude a confounder
   - C) We include a mediator
   - D) We include a collider

3. **Measurement error in a confounder causes:**
   - A) Upward bias
   - B) Downward bias (attenuation)
   - C) No bias
   - D) Inconsistency

4. **Including a post-treatment variable:**
   - A) Reduces bias
   - B) Increases bias
   - C) Has no effect
   - D) Depends on the situation

5. **The coefficient on treatment in multiple regression is:**
   - A) The correlation between T and Y
   - B) The partial effect of T on Y, holding X constant
   - C) The total effect of T on Y
   - D) The direct effect of T on Y

### Short Answer

6. **Explain why regression is sometimes called "adjustment for confounders."**

7. **What conditions must hold for regression to estimate causal effects?**

8. **How can you detect functional form misspecification?**

9. **Why might adding more variables to a regression increase bias?**

10. **Describe the difference between statistical and practical significance in regression.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Estimation of causal effects**](https://www.youtube.com/watch?v=YzcOYU-s2t4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=42)

Moves from identification to regression and flexible estimation.

**Active-viewing prompt:** Mark the point where a statistical model begins relying on a causal identification assumption.
```

---

## 7.10 Summary

1. **Regression** can estimate causal effects under unconfoundedness
2. **Multiple regression** controls for observed confounders
3. **Omitted variable bias** is the main threat
4. **Functional form** matters for correct estimation
5. **Diagnostics** are essential for validating regression models

---

## 7.11 Further Reading

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.


---

## Worked Examples

### Example 1: Medicine — Adjusting for Severity

A study compares two cancer treatments. Treatment A is given to sicker patients. A naive comparison shows Treatment A has worse outcomes. But after adjusting for cancer stage, age, and comorbidities, Treatment A is actually superior. The regression model is:

```
outcome ~ treatment + stage + age + comorbidities
```

The coefficient on treatment is the causal effect *conditional on* the covariates being held fixed.

### Example 2: Economics — Wage Equation

Estimating the return to education:

```
log(wage) = beta_0 + beta_1(education) + beta_2(experience) + beta_3(experience^2) + epsilon
```

**Problem**: Ability is omitted. If ability correlates with both education and wages, beta_1 is biased upward. This is why economists use instrumental variables.

### Example 3: Technology — User Engagement

A platform wants to know if a new feature increases engagement. They regress engagement on feature usage, controlling for user demographics and prior engagement:

```
engagement ~ feature_used + days_active + prior_engagement + demographics
```

**Omitted variable**: User motivation. Motivated users both adopt the feature faster and engage more. Without controlling for motivation, the feature effect is overestimated.

### Example 4: Policy — Class Size Effects

Regressing test scores on class size, controlling for school funding, teacher quality, and student demographics. School funding and teacher quality are correlated with class size (better-funded schools have smaller classes). Omitting either biases the class size coefficient.

---

## Diagnostics: Regression Assumptions

### The Four Key Assumptions

1. **Linearity**: The true relationship is linear
2. **Exogeneity**: E[epsilon|X] = 0 — no omitted variable bias
3. **Homoscedasticity**: Constant error variance
4. **No perfect multicollinearity**: No variable is a perfect linear combination of others

### Testing with Code

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_diagnostics(X, y, feature_names):
    """Run comprehensive regression diagnostics."""
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Heteroscedasticity test
    bp_test = het_breuschpagan(model.resid, sm.add_constant(X))
    print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")

    # VIF for multicollinearity
    for i, name in enumerate(feature_names):
        vif = variance_inflation_factor(X, i)
        if vif > 10:
            print(f"WARNING: {name} has high VIF ({vif:.1f})")

    return model
```

---

## Interpretation Workshop

### Common Misinterpretations

- "The coefficient is significant, therefore it is causal" — WRONG. Significance does not equal causality.
- "After controlling for X, the effect is Y" — Only causal if X is sufficient for confounding control.
- "The R-squared is low, so the model is bad" — Low R-squared is fine for causal inference; we care about the coefficient, not prediction.

### Coefficient Magnitude

Always ask: Is this effect size meaningful? A statistically significant coefficient of 0.001 may be real but trivially small.

---

## Practical Application

### Choosing Covariates: The DAG Approach

1. Draw the DAG for your specific problem
2. Identify confounders: Variables that cause both treatment and outcome
3. Include all confounders in the regression
4. Exclude colliders and descendants of the treatment
5. Consider precision variables: Variables that reduce residual variance without introducing bias

### The Table 2 Problem

Many researchers include all baseline covariates in their outcome regression. This is problematic if some covariates are affected by the treatment, are colliders, or are mediators (should not be controlled for when estimating total effects).

**Better approach**: Pre-specify covariates based on DAG reasoning.

---

## Limitations

- **Model misspecification**: Linearity assumption is often wrong
- **Omitted variables**: Cannot control for unmeasured confounders
- **Extrapolation**: Regression adjusts for observed confounders, not unobserved ones
- **Simpson's paradox**: Aggregated data can show opposite trends from disaggregated data

---

## Exercises

1. **DAG to regression**: Given a DAG with confounders C1, C2 and a precision variable P, write the regression equation. Which variables should be included?

2. **Diagnostics**: Run the regression diagnostics code on a dataset of your choice. What do you find?

3. **Omitted variable bias**: Simulate a scenario where omitting variable Z biases the treatment effect. Show that including Z recovers the true effect.

4. **Simpson's paradox**: Create a dataset where the overall correlation between X and Y is positive, but within every subgroup it is negative.

---

## Projects

### Project 1: Sensitivity to Covariate Selection
Take a dataset and estimate a causal effect using: no covariates, all available covariates, DAG-selected covariates, and data-driven covariates (LASSO). Compare estimates.

### Project 2: Replicate a Published Study
Find a study that uses OLS for causal inference. Identify the key regression, assess the DAG assumptions, test for heteroscedasticity and multicollinearity, and write a limitations section.
