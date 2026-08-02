---
title: "Lesson 22: Decision Theory and Causal Inference"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 22: Decision Theory and Causal Inference

## Opening Story: Personalized Treatment Rules

A doctor has a patient with high cholesterol. Should they prescribe statins? The decision depends on the patient's risk factors, potential side effects, and the expected benefit. This is a decision-theoretic problem: what treatment should we choose to maximize expected utility?

Decision theory provides the framework for optimal treatment assignment, connecting causal inference to actionable recommendations.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define optimal treatment rules
2. Explain the value of information
3. Implement policy learning
4. Conduct cost-benefit analysis
5. Connect causal inference to decision-making

---

## 22.1 Optimal Treatment Rules

### Definition

A treatment rule $\delta(X)$ assigns treatment based on observed covariates:

$$\delta(X) = \arg\max_{a \in \{0,1\}} E[Y(a) | X]$$

### Value of a Rule

The value of a treatment rule is the expected outcome if we follow it:

$$V(\delta) = E[Y^{\delta(X)}]$$

---

## 22.2 Policy Learning

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 1000
p = 5

# Generate data
X = np.random.normal(0, 1, (n, p))
T = np.random.binomial(1, 0.5, n)
Y = 2 * T + X @ np.ones(p) + np.random.normal(0, 1, n)

# Heterogeneous effects
true_effect = 2 + X[:, 0] - X[:, 1]

# Optimal rule: treat if effect > 0
optimal_rule = (true_effect > 0).astype(int)

# Value of optimal rule
value_optimal = Y[optimal_rule == 1].mean() * optimal_rule.mean() + \
                Y[optimal_rule == 0].mean() * (1 - optimal_rule).mean()

# Value of treat-all
value_all = Y.mean()

# Value of treat-none
value_none = Y[T == 0].mean()

print(f"Value of treat-all: {value_all:.3f}")
print(f"Value of treat-none: {value_none:.3f}")
print(f"Value of optimal rule: {value_optimal:.3f}")

# Learn a policy using treatment effects
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor

x_learner = XLearner(models=RandomForestRegressor(n_estimators=100, random_state=42))
x_learner.fit(Y, T, X=X)
te_pred = x_learner.effect(X)

# Learned rule: treat if predicted effect > 0
learned_rule = (te_pred > 0).astype(int)
value_learned = Y[learned_rule == 1].mean() * learned_rule.mean() + \
                Y[learned_rule == 0].mean() * (1 - learned_rule).mean()

print(f"Value of learned rule: {value_learned:.3f}")
```

---

## 22.3 Cost-Benefit Analysis

```python
# Include costs of treatment
cost_treatment = 100
benefit_treatment = 500  # Monetary value of health improvement

# Net benefit
def net_benefit(rule, Y, cost):
    treated = rule.sum()
    benefit = Y[rule == 1].sum() - cost * treated
    return benefit

# Compare policies
print(f"\nNet benefit (optimal): {net_benefit(optimal_rule, Y, cost_treatment):.2f}")
print(f"Net benefit (learned): {net_benefit(learned_rule, Y, cost_treatment):.2f}")
print(f"Net benefit (treat all): {net_benefit(np.ones(n), Y, cost_treatment):.2f}")
```

---

## 22.4 Common Mistakes

1. **Ignoring heterogeneity**: Treat-all is suboptimal when effects vary
2. **Overfitting policies**: Use cross-fitting for valid estimation
3. **Ignoring costs**: Benefits must outweigh costs
4. **Short-term vs long-term**: Consider time horizons

---

## 22.5 Knowledge Check

### Multiple Choice

1. **An optimal treatment rule:**
   A) Treats everyone
   B) Treats no one
   C) Assigns treatment based on covariates
   D) Randomizes treatment

2. **Policy learning:**
   A) Estimates average effects
   B) Learns treatment assignment rules
   C) Tests causal hypotheses
   D) All of the above

3. **The value of a rule is:**
   A) The average treatment effect
   B) The expected outcome under the rule
   C) The probability of treatment
   D) The variance of outcomes

4. **Cost-benefit analysis:**
   A) Ignores costs
   B) Considers both costs and benefits
   C) Only considers benefits
   D) Only considers costs

5. **Cross-fitting in policy learning:**
   A) Increases bias
   B) Reduces overfitting
   C) Has no effect
   D) Speeds computation

### Short Answer

6. **Explain how causal inference informs treatment decisions.**

7. **What is the difference between treat-all and an optimal rule?**

8. **How do you evaluate a learned treatment rule?**

9. **Why is cross-fitting important for policy learning?**

10. **Give an example where personalized treatment rules would be valuable.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Treatment-effect heterogeneity and policy choice**](https://www.youtube.com/watch?v=YzcOYU-s2t4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=42)

Provides the conditional-effect estimates that decision rules consume.

**Active-viewing prompt:** Separate estimation of outcomes from the utility function used to choose an action.
```

---

## 22.6 Summary

1. **Optimal treatment rules** assign treatment based on covariates
2. **Policy learning** estimates these rules from data
3. **Value assessment** compares rules to benchmarks
4. **Cost-benefit analysis** incorporates real-world constraints
5. **Cross-fitting** ensures valid policy evaluation

---

## 22.7 Further Reading

- Robins, J.M. (2000). "Optimal Structural Nested Models for Optimal Sequential Decisions."
- Kitagawa, T. & Tetenov, A. (2018). "Who Should Be Treated? Optimal Treatment Assignment Rules." *Econometrics*.


---

## Worked Examples

### Example 1: Medical Decision Making

A patient must choose between surgery (high risk, high reward) and medication (low risk, moderate reward). Decision theory provides a framework for combining treatment effects with patient preferences.

### Example 2: Policy Cost-Benefit Analysis

A government evaluates a new policy by comparing expected benefits (reduced crime, improved health) to costs (implementation, enforcement). The optimal policy maximizes expected welfare.

### Example 3: Clinical Trial Design

Adaptive trial designs use decision theory to determine when to stop early for efficacy or futility, balancing the value of information against the cost of continued experimentation.

### Example 4: Personalized Treatment

A doctor chooses treatments based on patient characteristics, balancing expected outcomes against treatment costs and side effects. Optimal treatment rules come from decision-theoretic analysis.

---

## Diagnostics: Decision-Theoretic Framework

### Value of Information

```python
import numpy as np

def expected_value_of_perfect_information(outcomes, probabilities, costs):
    """Calculate EVPI for a decision problem."""

    # Expected value with perfect information
    ev_perfect = sum(max(outcomes[i]) * probabilities[i] for i in range(len(probabilities)))

    # Expected value without additional information
    ev_current = max(sum(outcomes[i][j] * probabilities[i] for i in range(len(probabilities)))
                     for j in range(len(outcomes[0])))

    evpi = ev_perfect - ev_current

    print(f"Expected Value with Perfect Information: {ev_perfect:.2f}")
    print(f"Expected Value without Information: {ev_current:.2f}")
    print(f"EVPI: {evpi:.2f}")

    return evpi
```

### Optimal Treatment Rules

```python
def optimal_treatment_rule(outcomes, costs, risk_aversion=0):
    """Find optimal treatment rule."""

    n_treatments = len(outcomes)
    n_patients = len(outcomes[0])

    # Expected utility for each treatment
    utilities = []
    for t in range(n_treatments):
        utility = np.mean(outcomes[t]) - risk_aversion * np.var(outcomes[t]) - costs[t]
        utilities.append(utility)

    optimal = np.argmax(utilities)

    print("Treatment Utilities:")
    for t in range(n_treatments):
        marker = " <-- OPTIMAL" if t == optimal else ""
        print(f"  Treatment {t}: {utilities[t]:.3f}{marker}")

    return optimal
```

---

## Interpretation Workshop

### Key Decision-Theoretic Concepts

- **Expected utility**: The average outcome weighted by probabilities
- **Risk aversion**: Preference for less variable outcomes
- **Value of information**: How much would perfect knowledge improve the decision?
- **Opportunity cost**: What is lost by not choosing the best alternative

### When Decision Theory Matters

- Resources are limited: Must choose which treatments to fund
- Uncertainty is high: Must balance expected outcomes against risk
- Multiple stakeholders: Must consider distributional effects
- Information is costly: Must decide when to stop gathering evidence

---

## Practical Application

### Bayesian Decision Theory

```python
def bayesian_decision(prior, likelihoods, costs, loss_function):
    """Make decision under uncertainty using Bayesian approach."""

    # Posterior probabilities
    posteriors = prior * likelihoods
    posteriors = posteriors / np.sum(posteriors)

    # Expected loss for each decision
    n_decisions = len(loss_function)
    expected_losses = []

    for d in range(n_decisions):
        el = np.sum(posteriors * loss_function[d])
        expected_losses.append(el)

    optimal = np.argmin(expected_losses)

    print("Expected Losses:")
    for d in range(n_decisions):
        marker = " <-- OPTIMAL" if d == optimal else ""
        print(f"  Decision {d}: {expected_losses[d]:.3f}{marker}")

    return optimal
```

### Cost-Effectiveness Analysis

```python
def incremental_cost_effectiveness(outcomes, costs):
    """Calculate incremental cost-effectiveness ratios."""

    # Sort by cost
    sorted_idx = np.argsort(costs)

    print("Incremental Cost-Effectiveness Ratios:")
    for i in range(1, len(sorted_idx)):
        icr = (costs[sorted_idx[i]] - costs[sorted_idx[i-1]]) / \
              (outcomes[sorted_idx[i]] - outcomes[sorted_idx[i-1]])
        print(f"  {sorted_idx[i]} vs {sorted_idx[i-1]}: {icr:.2f} per unit outcome")
```

---

## Limitations

- Requires quantifying preferences (utility functions)
- Sensitive to risk aversion assumptions
- May ignore distributional concerns
- Information value depends on accuracy of probability estimates

---

## Exercises

1. **EVPI calculation**: Calculate the expected value of perfect information for a medical decision problem.
2. **Optimal rule**: Derive the optimal treatment rule for a two-treatment problem with different risk profiles.
3. **Bayesian decision**: Make a decision under uncertainty using Bayesian updating.
4. **Cost-effectiveness**: Compare two treatments using incremental cost-effectiveness analysis.

---

## Projects

### Project 1: Clinical Trial Design
Design an adaptive clinical trial using decision theory to determine stopping rules.

### Project 2: Policy Optimization
Optimize a policy intervention using cost-benefit analysis and decision theory.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/22-decision-theory.ipynb)
- [Download the practice lab](../labs/lab22-decision-theory-practice.ipynb)
- [Download the lab solution](../solutions/lab22-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
