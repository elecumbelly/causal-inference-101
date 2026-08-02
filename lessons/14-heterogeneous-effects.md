---
title: "Lesson 14: Heterogeneous Treatment Effects"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 14: Heterogeneous Treatment Effects

## Opening Story: Personalized Medicine

A drug might work well for some patients but not others. Understanding who benefits most from treatment is crucial for personalized medicine and policy design.

Treatment effect heterogeneity—variation in treatment effects across individuals—is now a central focus of causal inference. Methods like CATE estimation and causal forests help us discover which subgroups benefit most.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define conditional average treatment effects (CATE)
2. Estimate CATE using causal forests
3. Identify subgroups with different treatment effects
4. Interpret heterogeneous treatment effects
5. Apply policy learning methods

---

## 14.1 Conditional Average Treatment Effects

### Definition

$$\tau(x) = E[Y(1) - Y(0) | X = x]$$

The treatment effect varies across subgroups defined by covariates $X$.

### Why It Matters

- **Policy targeting**: Focus resources on those who benefit most
- **Equity**: Understand if effects differ across demographics
- **Mechanism**: Discover why treatments work differently

---

## 14.2 Causal Forests

```python
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n = 2000
p = 5

# Generate data with heterogeneous effects
X = np.random.normal(0, 1, (n, p))
T = np.random.binomial(1, 0.5, n)

# Treatment effect depends on X[:, 0]
true_effect = 2 * X[:, 0]
Y = 10 + X @ np.ones(p) + true_effect * T + np.random.normal(0, 1, n)

# Fit causal forest
causal_forest = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
    n_estimators=100,
    random_state=42
)

causal_forest.fit(Y, T, X=X)

# Estimate individual treatment effects
te_pred = causal_forest.effect(X)

# Compare to true effects
correlation = np.corrcoef(true_effect, te_pred)[0, 1]
print(f"Correlation between true and predicted effects: {correlation:.3f}")

# Feature importance
importances = causal_forest.feature_importances_
print(f"\nFeature importances: {importances.round(3)}")
print(f"True importance: X[:, 0] should be highest")
```

---

## 14.3 Policy Learning

```python
# Optimal treatment assignment
# Assign treatment to those with highest predicted effects
threshold = np.median(te_pred)
policy = (te_pred > threshold).astype(int)

# Value of policy vs. always treat
value_always = true_effect.mean()
value_policy = true_effect[policy == 1].mean() * policy.mean() + \
               0 * (1 - policy).mean()

print(f"Value of always treating: {value_always:.3f}")
print(f"Value of policy: {value_policy:.3f}")
```

---

## 14.4 Common Mistakes

1. **Overfitting**: Use cross-fitting
2. **Multiple testing**: Adjust for multiple comparisons
3. **Interpretation**: CATE is correlational, not necessarily causal
4. **Data splitting**: Use sample splitting for honest estimation

---

## 14.5 Knowledge Check

### Multiple Choice

1. **CATE is:**
   A) Average treatment effect for everyone
   B) Treatment effect for a specific subgroup
   C) Total treatment effect
   D) Direct treatment effect

2. **Causal forests:**
   A) Are random forests for outcomes
   B) Estimate heterogeneous treatment effects
   C) Are always better than regression
   D) Don't require assumptions

3. **Policy learning:**
   A) Determines who should be treated
   B) Estimates average effects
   C) Tests for heterogeneity
   D) All of the above

4. **Heterogeneous effects are important for:**
   A) Personalized medicine
   B) Policy targeting
   C) Understanding mechanisms
   D) All of the above

5. **Cross-fitting is used to:**
   A) Increase bias
   B) Reduce overfitting
   C) Speed up computation
   D) Increase variance

### Short Answer

6. **Explain why treatment effect heterogeneity matters for policy.**

7. **How do causal forests differ from random forests?**

8. **What is the role of cross-fitting in causal forest estimation?**

9. **How can you identify subgroups with high treatment effects?**

10. **Give an example where heterogeneous effects would change policy recommendations.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Estimating heterogeneous treatment effects**](https://www.youtube.com/watch?v=YzcOYU-s2t4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=42)

Introduces conditional effects and the estimation choices behind personalization.

**Active-viewing prompt:** Ask whether discovered subgroups were prespecified, honest, and supported by overlap.
```

---

## 14.6 Summary

1. **Heterogeneous effects** vary across subgroups
2. **CATE** is the key quantity of interest
3. **Causal forests** estimate CATE nonparametrically
4. **Policy learning** uses CATE for optimal treatment assignment
5. **Cross-fitting** ensures valid inference

---

## 14.7 Further Reading

- Athey, S. & Imbens, G.W. (2016). "Recursive Partitioning for Heterogeneous Causal Effects." *PNAS*.
- Wager, S. & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA*.


---

## Worked Examples

### Example 1: Medicine — Drug Effect by Age Group

A drug may work better for younger patients than older ones. CATE estimation identifies which patients benefit most, enabling personalized treatment recommendations.

### Example 2: Economics — Job Training by Education Level

A job training program may have larger effects for high school graduates than for college graduates. Policy targeting depends on this heterogeneity — resources should go where they have the largest impact.

### Example 3: Technology — Feature Effect by User Segment

A new feature may increase engagement for power users but not casual users. Personalized rollout depends on CATE estimation to avoid wasting resources on non-responsive segments.

### Example 4: Policy — Education Intervention by Socioeconomic Status

A tutoring program may have larger effects for low-SES students. Equity considerations require understanding effect heterogeneity to ensure fair resource allocation.

---

## Diagnostics: CATE Estimation

### Meta-Learners

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

class XLearner:
    """X-Learner for CATE estimation."""

    def __init__(self):
        self.models = {}

    def fit(self, X, T, Y):
        """Fit X-Learner."""
        treated = T == 1
        control = T == 0

        # Stage 1: Outcome models for treatment and control
        self.models['outcome_treated'] = GradientBoostingRegressor()
        self.models['outcome_control'] = GradientBoostingRegressor()

        self.models['outcome_treated'].fit(X[treated], Y[treated])
        self.models['outcome_control'].fit(X[control], Y[control])

        # Stage 2: Impute treatment effects
        tau_treated = Y[treated] - self.models['outcome_control'].predict(X[treated])
        tau_control = self.models['outcome_treated'].predict(X[control]) - Y[control]

        # Stage 3: Propensity score model
        self.models['propensity'] = RandomForestClassifier()
        self.models['propensity'].fit(X, T)

        # Stage 4: CATE models
        self.models['cate_treated'] = GradientBoostingRegressor()
        self.models['cate_control'] = GradientBoostingRegressor()

        self.models['cate_treated'].fit(X[treated], tau_treated)
        self.models['cate_control'].fit(X[control], tau_control)

        return self

    def predict(self, X):
        """Predict CATE."""
        e = self.models['propensity'].predict_proba(X)[:, 1]
        tau_treated = self.models['cate_treated'].predict(X)
        tau_control = self.models['cate_control'].predict(X)

        tau = e * tau_control + (1 - e) * tau_treated

        return tau
```

### Validation of CATE Estimates

```python
def validate_cate(X, T, Y, cate_estimates, n_groups=5):
    """Validate CATE estimates by comparing within groups."""

    sorted_idx = np.argsort(cate_estimates)
    groups = np.array_split(sorted_idx, n_groups)

    print("CATE Validation by Predicted Effect Group:")
    print("=" * 50)

    for i, group in enumerate(groups):
        group_tau = cate_estimates[group]
        mean_tau = np.mean(group_tau)

        treated = T[group] == 1
        control = T[group] == 0

        if treated.sum() > 0 and control.sum() > 0:
            y_treated = Y[group][treated].mean()
            y_control = Y[group][control].mean()
            observed_effect = y_treated - y_control

            print(f"Group {i+1}: Predicted CATE = {mean_tau:.3f}, Observed effect = {observed_effect:.3f}")
```

---

## Interpretation Workshop

### When Heterogeneity Matters

- **Policy targeting**: Who should receive the intervention?
- **Equity analysis**: Does the intervention benefit disadvantaged groups?
- **Personalization**: Can we tailor treatment to individual characteristics?
- **External validity**: Will the effect generalize to new populations?

### Interpreting CATE Plots

- Positive CATE: Subgroup benefits from treatment
- Negative CATE: Subgroup is harmed by treatment
- Zero CATE: No effect for this subgroup
- Wide confidence intervals: Uncertain about effect in this subgroup

---

## Practical Application

### Decision Trees for CATE

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

def cate_tree(X, T, Y, feature_names=None):
    """Use decision tree to identify subgroups with different effects."""

    treated = T == 1
    control = T == 0

    model_treated = GradientBoostingRegressor()
    model_control = GradientBoostingRegressor()

    model_treated.fit(X[treated], Y[treated])
    model_control.fit(X[control], Y[control])

    tau_hat = model_treated.predict(X) - model_control.predict(X)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50)
    tree.fit(X, tau_hat)

    plt.figure(figsize=(15, 8))
    plot_tree(tree, feature_names=feature_names, filled=True, fontsize=10, proportion=True)
    plt.title("Subgroups with Different Treatment Effects")
    plt.tight_layout()

    return tree, tau_hat
```

---

## Limitations

- Sample size: CATE estimation requires large samples
- Overfitting: Easy to find spurious heterogeneity
- Multiple testing: Many subgroup comparisons inflate false positives
- External validity: CATE estimates may not generalize

---

## Exercises

1. **Implement CATE estimation**: Use the X-Learner to estimate heterogeneous treatment effects. How do results compare to a simple T-Learner?
2. **Subgroup analysis**: Use decision trees to identify subgroups. Are the subgroups interpretable?
3. **Validation**: How would you validate CATE estimates out-of-sample?
4. **Policy implication**: If you found that the treatment works better for men than women, what policy implications would this have?

---

## Projects

### Project 1: CATE Comparison Study
Compare T-Learner, S-Learner, X-Learner, and R-Learner on simulated data with known heterogeneity.

### Project 2: HTE in Practice
Apply CATE estimation to a real dataset. Identify which subgroups benefit most and least from the treatment.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/14-heterogeneous-effects.ipynb)
- [Download the practice lab](../labs/lab14-heterogeneous-effects-practice.ipynb)
- [Download the lab solution](../solutions/lab14-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
