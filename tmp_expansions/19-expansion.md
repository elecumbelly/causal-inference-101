

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
