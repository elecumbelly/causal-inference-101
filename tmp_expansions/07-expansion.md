

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
