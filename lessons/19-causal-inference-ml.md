---
title: "Lesson 19: Causal Inference in Machine Learning"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 19: Causal Inference in Machine Learning

## Opening Story: Recommendation Systems

Netflix recommends movies based on what you've watched. But is this causal? If you watch a thriller because Netflix recommended it, and then you get more thriller recommendations, is Netflix actually causing your preferences?

Machine learning is excellent at prediction but struggles with causation. Causal inference in ML bridges this gap, enabling algorithms that understand not just what happened, but why.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain why ML needs causal inference
2. Implement causal forests
3. Use double machine learning
4. Apply meta-learners for treatment effect estimation
5. Recognize limitations of standard ML for causal questions

---

## 19.1 Why ML Needs Causation

### Prediction vs. Causation

- **Prediction**: What will happen given observed features?
- **Causation**: What will happen if we intervene?

Standard ML optimizes prediction accuracy. Causal ML optimizes treatment effect estimation.

### The Problem with Correlations

ML algorithms learn correlations, which can be:
- Spurious (due to confounding)
- Reverse (effect predicted as cause)
- Non-causal (due to selection)

---

## 19.2 Causal Forests

```python
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n = 2000
p = 10

# Generate data
X = np.random.normal(0, 1, (n, p))
T = np.random.binomial(1, 0.5, n)

# Heterogeneous treatment effect
true_effect = 2 * X[:, 0] + X[:, 1] ** 2
Y = true_effect * T + X @ np.ones(p) + np.random.normal(0, 1, n)

# Fit causal forest
causal_forest = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
    n_estimators=100,
    random_state=42
)

causal_forest.fit(Y, T, X=X)

# Predict treatment effects
te_pred = causal_forest.effect(X)

# Evaluate
correlation = np.corrcoef(true_effect, te_pred)[0, 1]
mse = np.mean((true_effect - te_pred) ** 2)

print(f"Correlation between true and predicted effects: {correlation:.3f}")
print(f"MSE: {mse:.3f}")
```

---

## 19.3 Double Machine Learning

```python
from econml.dml import LinearDML

# Double ML for average treatment effect
dml = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
    random_state=42
)

dml.fit(Y, T, X=X)

# Average treatment effect
ate = dml.ate(X)
ate_interval = dml.ate_interval(X, alpha=0.05)

print(f"ATE estimate: {ate:.3f}")
print(f"95% CI: [{ate_interval[0]:.3f}, {ate_interval[1]:.3f}]")
print(f"True ATE: {true_effect.mean():.3f}")
```

---

## 19.4 Meta-Learners

```python
from econml.metalearners import TLearner, SLearner, XLearner

# S-Learner
s_learner = SLearner(
    overall_model=RandomForestRegressor(n_estimators=100, random_state=42)
)
s_learner.fit(Y, T, X=X)
te_s = s_learner.effect(X)

# T-Learner
t_learner = TLearner(
    models=RandomForestRegressor(n_estimators=100, random_state=42)
)
t_learner.fit(Y, T, X=X)
te_t = t_learner.effect(X)

# X-Learner
x_learner = XLearner(
    models=RandomForestRegressor(n_estimators=100, random_state=42)
)
x_learner.fit(Y, T, X=X)
te_x = x_learner.effect(X)

print("S-Learner correlation:", np.corrcoef(true_effect, te_s)[0, 1].round(3))
print("T-Learner correlation:", np.corrcoef(true_effect, te_t)[0, 1].round(3))
print("X-Learner correlation:", np.corrcoef(true_effect, te_x)[0, 1].round(3))
```

---

## 19.5 Common Mistakes

1. **Using ML for causal inference directly**: Standard ML gives biased treatment effect estimates
2. **Ignoring confounding**: ML doesn't handle confounding by default
3. **Overfitting**: Use cross-fitting and sample splitting
4. **Wrong evaluation metrics**: Use causal metrics, not prediction metrics

---

## 19.6 Knowledge Check

### Multiple Choice

1. **Standard ML:**
   A) Handles causation naturally
   B) Optimizes prediction accuracy
   C) Estimates treatment effects
   D) All of the above

2. **Causal forests estimate:**
   A) Average treatment effects
   B) Conditional average treatment effects
   C) Total effects
   D) Direct effects

3. **Double ML:**
   A) Requires two datasets
   B) Uses cross-fitting to reduce bias
   C) Doubles the sample size
   D) Is always better than single ML

4. **Meta-learners:**
   A) Are always identical
   B) Handle heterogeneous effects
   C) Don't need any assumptions
   D) Are only for experiments

5. **Cross-fitting is used to:**
   A) Increase overfitting
   B) Reduce overfitting
   C) Speed up computation
   D) Increase variance

### Short Answer

6. **Why can't we use standard ML for causal inference?**

7. **What is the role of cross-fitting in causal ML?**

8. **How do meta-learners differ from each other?**

9. **When would you use causal forests vs. double ML?**

10. **What are the limitations of causal ML methods?**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Estimation and machine learning for causal effects**](https://www.youtube.com/watch?v=YzcOYU-s2t4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=42)

Explains why predictive flexibility helps only after the causal target is identified.

**Active-viewing prompt:** Separate nuisance-function prediction quality from bias in the causal estimand.
```

---

## 19.7 Summary

1. **ML needs causation** for treatment effect estimation
2. **Causal forests** estimate heterogeneous effects
3. **Double ML** uses cross-fitting for valid inference
4. **Meta-learners** provide flexible effect estimation
5. **Cross-fitting** is essential for validity

---

## 19.8 Further Reading

- Athey, S. & Imbens, G.W. (2019). "Machine Learning Methods That Economists Should Know About." *Annual Review of Economics*.
- Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning." *Econometrics Journal*.


---

## Worked Examples

### Example 1: Causal Forest

Estimating heterogeneous treatment effects using generalized random forests. The forest adapts to the local structure of the CATE function, providing flexible estimation without parametric assumptions.

### Example 2: Double Machine Learning

Using LASSO to select confounders, then estimating the treatment effect with debiased inference. This combines ML flexibility with causal validity by orthogonalizing the treatment effect from the nuisance parameters.

### Example 3: Targeted Maximum Likelihood Estimation (TMLE)

A semiparametric efficient estimator that uses machine learning for nuisance parameter estimation while maintaining valid inference for the causal effect.

### Example 4: Neural Network CATE

Using neural networks to estimate CATE with architectures like DragonNet, which shares representations between treatment and outcome models to improve efficiency.

---

## Diagnostics: ML for Causal Inference

### Double/Debiased Machine Learning

```python
import numpy as np
from sklearn.linear_model import LassoCV
from econml.dml import LinearDML, CausalForestDML

def double_ml_example(Y, T, X, W):
    """Double/Debiased Machine Learning example."""

    # Method 1: Linear DML with LASSO
    dml = LinearDML(
        model_y=LassoCV(),
        model_t=LassoCV(),
        discrete_treatment=False
    )
    dml.fit(Y, T, X=X, W=W)

    ate = dml.ate(X)
    ate_interval = dml.ate_interval(X)

    print(f"ATE via DML: {ate:.3f}")
    print(f"95% CI: [{ate_interval[0]:.3f}, {ate_interval[1]:.3f}]")

    # Method 2: Causal Forest for CATE
    forest = CausalForestDML(
        model_y=LassoCV(),
        model_t=LassoCV(),
        n_estimators=100
    )
    forest.fit(Y, T, X=X, W=W)

    cate = forest.effect(X)

    print(f"CATE distribution:")
    print(f"  Mean: {np.mean(cate):.3f}")
    print(f"  Std: {np.std(cate):.3f}")

    return dml, forest
```

### TMLE Implementation

```python
def tmle_estimator(Y, T, X, W):
    """Targeted Maximum Likelihood Estimation."""

    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize

    # Step 1: Initial outcome model
    Q_model = LogisticRegression(max_iter=1000)
    Q_model.fit(np.column_stack([T, W]), Y)

    Q1 = Q_model.predict_proba(np.column_stack([np.ones(len(W)), W]))[:, 1]
    Q0 = Q_model.predict_proba(np.column_stack([np.zeros(len(W)), W]))[:, 1]

    # Step 2: Propensity score model
    g_model = LogisticRegression(max_iter=1000)
    g_model.fit(W, T)
    g1 = g_model.predict_proba(W)[:, 1]

    # Step 3: Targeting step
    H1 = T / g1
    H0 = (1 - T) / (1 - g1)

    def neg_loglik(epsilon):
        Q1_star = Q1 * np.exp(epsilon * H1)
        Q0_star = Q0 * np.exp(epsilon * H0)
        psi = np.mean(Q1_star - Q0_star)
        return -np.sum(Y * np.log(Q1_star / Q0_star + 1e-10))

    result = minimize(neg_loglik, 0)
    epsilon = result.x[0]

    # Step 4: Compute ATE
    Q1_star = Q1 * np.exp(epsilon * H1)
    Q0_star = Q0 * np.exp(epsilon * H0)
    ate = np.mean(Q1_star - Q0_star)

    return ate
```

---

## Interpretation Workshop

### When ML Helps Causal Inference

- High-dimensional confounders: When the number of confounders is large
- Nonlinear relationships: When the outcome model is complex
- Variable selection: When you need to identify which confounders matter
- Heterogeneous effects: When CATE varies smoothly with covariates

### When ML Hurts Causal Inference

- Overfitting: Can lead to biased estimates
- Regularization bias: LASSO may exclude important confounders
- Interpretability: Complex models are harder to validate
- Inference: Standard errors may be invalid without debiasing

---

## Practical Application

### Causal Forest for HTE

```python
from econml.dml import CausalForestDML

def causal_forest_hte(Y, T, X, W):
    """Estimate HTE using Causal Forest."""

    forest = CausalForestDML(
        model_y='auto',
        model_t='auto',
        n_estimators=2000,
        min_samples_leaf=10,
        max_depth=10
    )

    forest.fit(Y, T, X=X, W=W)

    cate = forest.effect(X)
    cate_interval = forest.effect_interval(X)
    var_importance = forest.feature_importances_

    return {
        'cate': cate,
        'interval': cate_interval,
        'importance': var_importance
    }
```

---

## Limitations

- Data hungry: ML methods require large samples
- Regularization bias: Can bias treatment effect estimates
- Black box: Hard to validate assumptions
- Computational cost: Training many models is expensive

---

## Exercises

1. **DML implementation**: Use DoubleML to estimate a treatment effect. Compare to OLS. How do results differ?
2. **Causal forest**: Estimate CATE using causal forest. How does it compare to subgroup analysis?
3. **TMLE**: Implement TMLE manually. How does it compare to DML?
4. **Model selection**: Compare different ML models (LASSO, random forest, neural network) for the nuisance parameters.

---

## Projects

### Project 1: ML Causal Inference Comparison
Compare DML, TMLE, and causal forest on simulated data with known CATE.

### Project 2: Deep Learning for CATE
Implement DragonNet or TARNet for CATE estimation using PyTorch.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/19-causal-inference-ml.ipynb)
- [Download the practice lab](../labs/lab19-causal-inference-ml-practice.ipynb)
- [Download the lab solution](../solutions/lab19-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
