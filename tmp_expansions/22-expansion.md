

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
