---
title: "Lesson 23: Fairness and Causal Inference"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 23: Fairness and Causal Inference

## Opening Story: Algorithmic Bias

In 2019, a study found that a healthcare algorithm used on millions of patients was biased against Black patients. The algorithm used healthcare costs as a proxy for health needs, but because Black patients had less access to healthcare, they had lower costs even when equally sick.

This is a causal problem: the algorithm confounded cost with need, leading to unfair treatment. Causal inference provides tools to understand and address algorithmic bias.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define fairness criteria in causal terms
2. Explain the difference between observational and causal fairness
3. Implement fairness-aware algorithms
4. Conduct causal audits of algorithms
5. Recognize the limitations of technical fairness solutions

---

## 23.1 Fairness Criteria

### Observational Fairness

- **Demographic parity**: $P(\hat{Y} = 1 | A = 0) = P(\hat{Y} = 1 | A = 1)$
- **Equalized odds**: $P(\hat{Y} = 1 | Y = y, A = 0) = P(\hat{Y} = 1 | Y = y, A = 1)$

### Causal Fairness

- **Counterfactual fairness**: $\hat{Y}_{A \leftarrow a} = \hat{Y}_{A \leftarrow a'}$ for all $a, a'$
- **Path-specific fairness**: No unfair discrimination through prohibited paths

---

## 23.2 Counterfactual Fairness

```python
import numpy as np
import pandas as pd
import pymc as pm

np.random.seed(42)
n = 1000

# Generate data
A = np.random.binomial(1, 0.5, n)  # Protected attribute
Z = np.random.normal(0, 1, n)  # Sensitive factor
X = 0.5 * A + 0.5 * Z + np.random.normal(0, 1, n)  # Covariates
Y = 2 * X + 0.3 * Z + np.random.normal(0, 1, n)  # Outcome

# Counterfactual fairness test
# For each individual, compare:
# 1. What would Y be if A = 0?
# 2. What would Y be if A = 1?

# If these differ, the decision is not counterfactually fair

# Simplified: check if A directly affects Y
from sklearn.linear_model import LinearRegression

# Model with A
model_with = LinearRegression()
model_with.fit(np.column_stack([X, A]), Y)

# Model without A
model_without = LinearRegression()
model_without.fit(X.reshape(-1, 1), Y)

# If A has a direct effect, counterfactual fairness is violated
print(f"Effect of A on Y (controlling for X): {model_with.coef_[1]:.3f}")
print(f"If this is non-zero, counterfactual fairness may be violated")
```

---

## 23.3 Path-Specific Fairness

### The Idea

Different causal paths from protected attribute to outcome may be fair or unfair:

- **Direct discrimination**: A → Y (unfair)
- **Indirect discrimination**: A → X → Y (may be fair or unfair)
- **Mediation**: A → Z → Y (depends on Z)

### Implementation

```python
# Check for direct vs indirect effects
# Direct effect: A → Y
# Indirect effect: A → X → Y

# Mediation analysis to decompose effects
from sklearn.linear_model import LinearRegression

# Total effect
model_total = LinearRegression()
model_total.fit(A.reshape(-1, 1), Y)
total_effect = model_total.coef_[0]

# Direct effect (controlling for X)
model_direct = LinearRegression()
model_direct.fit(np.column_stack([A, X]), Y)
direct_effect = model_direct.coef_[0]

# Indirect effect
indirect_effect = total_effect - direct_effect

print(f"Total effect of A: {total_effect:.3f}")
print(f"Direct effect: {direct_effect:.3f}")
print(f"Indirect effect (through X): {indirect_effect:.3f}")
```

---

## 23.4 Common Mistakes

1. **Ignoring causal structure**: Observational fairness criteria can be misleading
2. **Proxy discrimination**: Using variables that encode protected attributes
3. **Feedback loops**: Algorithms can amplify existing biases
4. **Simplicity**: Fairness is context-dependent, not just technical

---

## 23.5 Knowledge Check

### Multiple Choice

1. **Counterfactual fairness requires:**
   A) Equal outcomes
   B) Equal treatment
   C) Same prediction regardless of protected attribute
   D) No discrimination

2. **Direct discrimination is:**
   A) A → Y
   B) A → X → Y
   C) X → Y
   D) Y → A

3. **Observational fairness criteria:**
   A) Always imply causal fairness
   B) Never imply causal fairness
   C) May or may not imply causal fairness
   D) Are sufficient for fairness

4. **Proxy discrimination occurs when:**
   A) A variable encodes protected attributes
   B) We measure the protected attribute directly
   C) We ignore the protected attribute
   D) We randomize treatment

5. **Fairness is:**
   A) A purely technical problem
   B) Always possible to achieve
   C) Context-dependent and contested
   D) Defined by algorithms

### Short Answer

6. **Explain the difference between observational and causal fairness.**

7. **What is proxy discrimination and why is it problematic?**

8. **How can causal inference help audit algorithms for bias?**

9. **Why might observational fairness criteria be insufficient?**

10. **Give an example where causal fairness analysis would be valuable.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Counterfactuals and path-specific effects**](https://www.youtube.com/watch?v=f8PEpthLlN4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=81)

Provides the counterfactual machinery behind individual and path-specific fairness criteria.

**Active-viewing prompt:** Ask which causal paths are declared permissible, and who gets to make that normative choice.
```

---

## 23.6 Summary

1. **Fairness** has both observational and causal definitions
2. **Counterfactual fairness** uses potential outcomes
3. **Path-specific fairness** decomposes effects by causal paths
4. **Causal audits** can identify sources of bias
5. **Context matters**—fairness is not just technical

---

## 23.7 Further Reading

- Kusner, M.J. et al. (2017). "From Parity to Preferences: The Case of Counterfactual Fairness." *NeurIPS*.
- Nabi, R. & Shpitser, I. (2018). "Fair Inference on Outcomes." *AAAI*.


---

## Worked Examples

### Example 1: Hiring Algorithms

An AI hiring tool trained on historical data may discriminate against women if past hiring decisions were biased. Fairness-aware causal inference attempts to correct for this while estimating the true effect of qualifications on hiring.

### Example 2: Criminal Justice Risk Assessment

Risk assessment tools used for bail decisions may perpetuate racial disparities. Causal fairness asks: would the prediction change if the defendant's race were different, holding all causally relevant factors constant?

### Example 3: Credit Scoring

Credit scoring models may use variables that are proxies for protected attributes. Fairness constraints ensure that similar applicants receive similar scores regardless of race or gender.

### Example 4: Healthcare Resource Allocation

Allocation algorithms for organ transplants or ICU beds must balance efficiency with equity. Causal fairness ensures that allocation rules do not discriminate based on protected characteristics.

---

## Diagnostics: Fairness Definitions

### Individual Fairness

```python
import numpy as np

def individual_fairness(model, data, protected_attr, distance_metric):
    """Check if similar individuals receive similar predictions."""

    predictions = model.predict(data)

    # Find similar pairs
    n = len(data)
    violations = 0

    for i in range(n):
        for j in range(i+1, n):
            dist = distance_metric(data[i], data[j])
            pred_diff = abs(predictions[i] - predictions[j])

            if pred_diff > dist:
                violations += 1

    print(f"Individual fairness violations: {violations}/{n*(n-1)//2}")

    return violations
```

### Group Fairness Metrics

```python
def group_fairness_metrics(y_true, y_pred, protected):
    """Calculate group fairness metrics."""

    groups = np.unique(protected)

    metrics = {}
    for group in groups:
        mask = protected == group
        metrics[group] = {
            'selection_rate': np.mean(y_pred[mask]),
            'true_positive_rate': np.mean(y_pred[mask & (y_true == 1)]),
            'false_positive_rate': np.mean(y_pred[mask & (y_true == 0)]),
        }

    print("Group Fairness Metrics:")
    for group in groups:
        print(f"  Group {group}:")
        for metric, value in metrics[group].items():
            print(f"    {metric}: {value:.3f}")

    # Disparate impact ratio
    if len(groups) == 2:
        ratio = metrics[groups[0]]['selection_rate'] / metrics[groups[1]]['selection_rate']
        print(f"\n  Disparate Impact Ratio: {ratio:.3f}")
        print(f"  (4/5 rule threshold: 0.8)")

    return metrics
```

### Counterfactual Fairness

```python
def counterfactual_fairness(model, individual, protected_attr, causal_model):
    """Check if prediction would change if protected attribute were different."""

    # Original prediction
    pred_original = model.predict(individual.reshape(1, -1))[0]

    # Counterfactual: change protected attribute
    counterfactual = individual.copy()
    counterfactual_idx = list(causal_model.variables).index(protected_attr)
    counterfactual[counterfactual_idx] = 1 - counterfactual[counterfactual_idx]

    pred_counterfactual = model.predict(counterfactual.reshape(1, -1))[0]

    diff = abs(pred_original - pred_counterfactual)

    print(f"Original prediction: {pred_original:.3f}")
    print(f"Counterfactual prediction: {pred_counterfactual:.3f}")
    print(f"Difference: {diff:.3f}")

    return diff
```

---

## Interpretation Workshop

### Trade-offs Between Fairness Definitions

- **Individual vs group**: Can satisfy group fairness while violating individual fairness
- **Calibration vs balance**: Different fairness criteria may be incompatible
- **Accuracy vs fairness**: Reducing bias may reduce overall accuracy

### When Fairness Analysis Matters

- Automated decisions affect people's lives (hiring, lending, criminal justice)
- Historical data may encode past discrimination
- Protected attributes may correlate with other features
- Stakeholders have different fairness preferences

---

## Practical Application

### Fairness-Aware Machine Learning

```python
def fairness_constrained_training(X, y, protected, constraint_type='demographic_parity'):
    """Train model with fairness constraints."""

    from sklearn.linear_model import LogisticRegression

    # Standard model
    model_standard = LogisticRegression(max_iter=1000)
    model_standard.fit(X, y)

    # Fairness-aware model (simplified)
    # In practice, use libraries like fairlearn or aif360

    # Evaluate fairness
    y_pred = model_standard.predict(X)
    group_fairness_metrics(y, y_pred, protected)

    return model_standard
```

### Post-Processing for Fairness

```python
def post_process_fairness(y_pred, protected, target_rate=0.5):
    """Adjust predictions to achieve fairness."""

    groups = np.unique(protected)
    adjusted = y_pred.copy()

    for group in groups:
        mask = protected == group
        current_rate = np.mean(y_pred[mask])

        if current_rate > target_rate:
            # Randomly flip some positive predictions
            n_flip = int((current_rate - target_rate) * mask.sum())
            flip_idx = np.where(mask & (y_pred == 1))[0][:n_flip]
            adjusted[flip_idx] = 0
        elif current_rate < target_rate:
            # Randomly flip some negative predictions
            n_flip = int((target_rate - current_rate) * mask.sum())
            flip_idx = np.where(mask & (y_pred == 0))[0][:n_flip]
            adjusted[flip_idx] = 1

    return adjusted
```

---

## Limitations

- Multiple incompatible fairness definitions
- Trade-off between fairness and accuracy
- Protected attributes may be unobserved
- Fairness criteria may not capture all ethical concerns

---

## Exercises

1. **Fairness metrics**: Calculate demographic parity, equalized odds, and calibration for a classification model.
2. **Trade-off analysis**: Show that equalized odds and calibration cannot both be satisfied simultaneously in some cases.
3. **Counterfactual fairness**: Implement counterfactual fairness for a simple model.
4. **Policy**: Design a hiring algorithm that satisfies demographic parity. What are the consequences?

---

## Projects

### Project 1: Fairness Audit
Conduct a fairness audit of a real-world algorithm using multiple fairness metrics.

### Project 2: Fairness-Aware Learning
Implement a fairness-aware machine learning algorithm and compare to unconstrained learning.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/23-fairness.ipynb)
- [Download the practice lab](../labs/lab23-fairness-practice.ipynb)
- [Download the lab solution](../solutions/lab23-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
